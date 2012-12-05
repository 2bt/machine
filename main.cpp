#include <signal.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
using namespace std;

#include <assert.h>

template <typename T>
class matrix {
	size_t h;
	size_t w;
	T* data;
public:
	matrix() : h(0), w(0), data(NULL) {}
	void init(size_t h, size_t w) {
		this->h = h;
		this->w = w;
		if (data) delete [] data;
		data = new T[w * h];
		zero();
	}
	void zero() {
		memset(data, 0, w * h * sizeof(T));
	}
	void fill(T f) {
		for(size_t i = 0; i < w * h; i++) data[i] = f;
	}
	~matrix() { delete [] data; }
	T* ptr() { return data; }
	T* operator[](size_t y) { return &data[y * w]; }
	size_t height() { return h; }
	size_t width() { return w; }
	void swap(matrix<T>& other) {
		T* t = other.data;
		other.data = data;
		data = t;
	}
	void normalize() {
		for (size_t x = 0; x < w; x++) {
			T s = h;
			for (size_t y = 0; y < h; y++) s += (*this)[y][x];
			s = 1 / s;
			for (size_t y = 0; y < h; y++) (*this)[y][x] = ((*this)[y][x] + 1) * s;
		}
	}
	void save(ofstream& file) {
		file.write((const char*) &h, sizeof(size_t));
		file.write((const char*) &w, sizeof(size_t));
		file.write((const char*) data, sizeof(T) * w * h);
	}
	void load(ifstream& file) {
		file.read((char*) &h, sizeof(size_t));
		file.read((char*) &w, sizeof(size_t));
		init(h, w);
		file.read((char*) data, sizeof(T) * w * h);
	}
};


size_t read_corpus(const string filename,
		map<string, size_t>& word2id,
		vector<string>& id2word,
		vector<vector<size_t>>& corpus) {


	ifstream file(filename.c_str());
	if (!file.is_open()) {
		cerr << "Error opening file " << filename << ".\n";
		exit(1);
	}

	string line;
	size_t maxlen = 0;
	while (getline(file, line)) {
		stringstream words(line);
		vector<size_t> sentence;
		string word;
		while (words >> word) {
			size_t id;
			if (word2id.count(word) == 0) {
				id = id2word.size();
				word2id[word] = id;
				id2word.push_back(word);
			} else id = word2id[word];
			sentence.push_back(id);
		}
		maxlen = max(maxlen, sentence.size());
		corpus.push_back(sentence);
	}
	return maxlen;
}


string					base;
string					e_lang;
string					f_lang;
map<string, size_t>		e_word2id;
map<string, size_t>		f_word2id;
vector<string>			e_id2word;
vector<string>			f_id2word;
matrix<float>			dict;
matrix<float>			langmodel;
matrix<float>			lenmodel;


void save() {
	cerr << "Saving...\n";
	string filename = base + ".meta-" + f_lang + "-to-" + e_lang;
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error opening file " << filename << ".\n";
		exit(1);
	}

	file << f_id2word.size() << "\n";
	for (auto s : f_id2word) file << s << "\n";
	file << e_id2word.size() << "\n";
	for (auto s : e_id2word) file << s << "\n";

	langmodel.save(file);
	lenmodel.save(file);
	dict.save(file);
}


void load() {
	cerr << "Loading...\n";
	string filename = base + ".meta-" + f_lang + "-to-" + e_lang;
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error opening file " << filename << ".\n";
		exit(1);
	}

	size_t len;
	string s;
	file >> len;
	for (size_t i = 0; i < len; i++) {
		file >> s;
		f_word2id[s] = i;
		f_id2word.push_back(s);
	}
	file >> len;
	for (size_t i = 0; i < len; i++) {
		file >> s;
		e_word2id[s] = i;
		e_id2word.push_back(s);
	}
	dict.init(f_id2word.size(), e_id2word.size());

	// binary for speed
	file.seekg(1, ios_base::cur); // skip newline
	langmodel.load(file);
	lenmodel.load(file);
	dict.load(file);
}


void leaving(int sig) {
	save();
	exit(0);
}


void train(int iterations) {
	cerr << "Reading corpora...\n";
	vector<vector<size_t>>	e_corpus;
	vector<vector<size_t>>	f_corpus;
	size_t f_msl = read_corpus(base + "." + f_lang, f_word2id, f_id2word, f_corpus);
	size_t e_msl = read_corpus(base + "." + e_lang, e_word2id, e_id2word, e_corpus);
	size_t corpus_size = e_corpus.size();
	if (corpus_size != f_corpus.size()) {
		cerr << "Corpora size differs.\n";
		exit(1);
	}


	// language model
	cerr << "Generating language model...\n";
	langmodel.init(e_id2word.size() + 1, e_id2word.size() + 1);
	for (vector<size_t>& sentence : e_corpus) {
		size_t prev_id = e_id2word.size();
		for (size_t id : sentence) {
			langmodel[id][prev_id]++;
			prev_id = id;
		}
		langmodel[e_id2word.size()][prev_id]++;
	}
	langmodel.normalize();


	// length model
	cerr << "Generating length model...\n";
	lenmodel.init(f_msl, e_msl);
	for (size_t l = 0; l < corpus_size; l++) {
		lenmodel[f_corpus[l].size() - 1][e_corpus[l].size() - 1]++;
	}
	lenmodel.normalize();


	// dictionary training
	matrix<float> c;
	dict.init(f_id2word.size(), e_id2word.size());
	c.init(f_id2word.size(), e_id2word.size());
	dict.fill(1);

	signal(SIGINT, leaving);
	cerr << "Training...\n";
	for (int count = 0; count < iterations; count++) {
		cerr << "Step " << count + 1 << "...\n";
		c.zero();
		for (size_t l = 0; l < corpus_size; l++) {
			for (size_t f : f_corpus[l]) {
				float s = 0;
				for (size_t e : e_corpus[l]) s += dict[f][e];
				s = 1 / s;
				for (size_t e : e_corpus[l]) c[f][e] += dict[f][e] * s;
			}
		}
		c.normalize();
		dict.swap(c);
	}
	save();
}


void lookup() {
	load();
	cerr << "Looking up...\n";
	string word;
	while (getline(cin, word)) {
		cout << word << "\n";
		if (!f_word2id.count(word)) {
			cout << "\tUnknown word.\n";
			continue;
		}
		size_t id = f_word2id[word];
		vector<size_t> top;
		float s = 0;
		for (size_t i = 0; i < e_id2word.size(); i++) {
			top.push_back(i);
			s += dict[id][i];
		}
		sort(top.begin(), top.end(), [&](size_t a, size_t b) {
			return dict[id][a] > dict[id][b];
		});
		for (size_t i = 0; i < 10; i++) {
			printf("\t%12.10f   %s\n", dict[id][top[i]] / s, e_id2word[top[i]].c_str());
		}
	}
}


float prob(const vector<size_t>& f_s, const vector<size_t>& e_s) {

	//if (e_s.size() < 1) return 0;

	float p = 1;
	for (size_t i = 1; i < e_s.size(); i++) {
		p *= langmodel[e_s[i]][e_s[i - 1]];
	}

	p /= pow(e_s.size() - 1 - (e_s.back() == e_id2word.size() - 1), f_s.size() - 2);


	//p /= 1 + pow(f_s.size() - e_s.size(), 2);
/*
	if (f_s.size() - 3 < lenmodel.height() || e_s.size() - 3 < lenmodel.width()) {
		p *= lenmodel[f_s.size() - 3][e_s.size() - 3];
	}
	else {
		return 1;
	}
*/

	for (size_t i = 1; i < f_s.size() - 1; i++) {
		float s = 0;
		for (size_t e : e_s) {
			if (e != e_id2word.size() - 1) {
				s += dict[f_s[i]][e];
			}
		}
		p *= s;
	}

	return p;
}


void print_sentence(vector<size_t>& s) {
	for (size_t i = 0; i < s.size(); i++) {
		if (i) cout << " ";
		cout << e_id2word[s[i]];
	}
	cout << "\n";
}


void decode() {
	load();
	cerr << "Decoding...\n";

	string line;
	while (getline(cin, line)) {
		stringstream words(line);
		vector<size_t> f_s;

		string word;
		while (words >> word) {
			size_t id;
			if (f_word2id.count(word) == 0) {
				cerr << "Unknown word (" << word << ").\n";
			} else id = f_word2id[word];
			f_s.push_back(id);
		}

		vector<vector<size_t>> H;
		vector<size_t> h;

		do {
			for (size_t e = 0; e < e_id2word.size(); e++) {

				h.push_back(e);
				H.push_back(h);
				h.pop_back();
				while (H.size() > 50) {

					size_t j = 0;
					float m = 9e9;
					for (size_t i = 0; i < H.size(); i++) {
						float s = prob(f_s, H[i]);
						if (s < m) {
							m = s;
							j = i;
						}
					}
					H.erase(H.begin() + j);
				}
			}

			size_t j = 0;
			float m = 0;
			for (size_t i = 0; i < H.size(); i++) {
				float s = prob(f_s, H[i]);
				//cout << s << ": "; print_sentence(H[i]);
				if (s > m) {
					m = s;
					j = i;
				}
			}
			assert(m > 0);
			h = H[j];
			H.erase(H.begin() + j);

		} while (h.back() != e_id2word.size() - 1);

		print_sentence(h);
	}
}


int main(int argc, char** argv) {
	if (argc < 5) {
		cout << "usage: " << argv[0] << " <base> <e> <f> <action> [iterations]\n";
		return 0;
	}
	base = argv[1];
	e_lang = argv[2];
	f_lang = argv[3];
	string action = argv[4];

	if (action == "train") train(argc > 5 ? atoi(argv[5]) : 100);
	else if (action == "lookup") lookup();
	else if (action == "decode") decode();
	else {
		cerr << "Invalid action.\n";
		return 1;
	}
	return 0;
}


