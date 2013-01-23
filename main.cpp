#include <signal.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <utility>

using namespace std;

#include <assert.h>


typedef vector<size_t> sentence_t;



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
			//for (size_t y = 0; y < h; y++) (*this)[y][x] = (*this)[y][x] * s; // no smooth
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
		vector<sentence_t>& corpus) {

	ifstream file(filename.c_str());
	if (!file.is_open()) {
		cerr << "Error opening file " << filename << ".\n";
		exit(1);
	}

	string line;
	size_t maxlen = 0;
	while (getline(file, line)) {
		stringstream words(line);
		sentence_t sentence;
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
matrix<double>			dict;
matrix<double>			langmodel;
matrix<double>			lenmodel;


void save() {
	cerr << "Saving...\n";
	string filename = base + ".meta-" + f_lang + "-to-" + e_lang;
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error opening file " << filename << ".\n";
		exit(1);
	}

	size_t len = f_id2word.size();
	file.write((const char*) &len, sizeof(size_t));
	for (auto s : f_id2word) file << s << "\n";
	len = e_id2word.size();
	file.write((const char*) &len, sizeof(size_t));
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
	file.read((char*) &len, sizeof(size_t));
	for (size_t i = 0; i < len; i++) {
		file >> s;
		f_word2id[s] = i;
		f_id2word.push_back(s);
	}
	file.get();
	file.read((char*) &len, sizeof(size_t));
	for (size_t i = 0; i < len; i++) {
		file >> s;
		e_word2id[s] = i;
		e_id2word.push_back(s);
	}
	dict.init(f_id2word.size(), e_id2word.size());

	// binary for speed
	file.get();
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
	vector<sentence_t>	e_corpus;
	vector<sentence_t>	f_corpus;
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
	for (const sentence_t& sentence : e_corpus) {
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
	matrix<double> c;
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
				double s = 0;
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
		sentence_t top;
		double s = 0;
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


void print_sentence(const sentence_t& s, const vector<string>& id2word) {
	for (size_t i = 0; i < s.size(); i++) {
		if (i) cout << " ";
		cout << id2word[s[i]];
	}
	cout << "\n";
}


// <STACK-DECODING>
double rate_heuristic(const sentence_t& e_s, const sentence_t& f_s) {
	double p = 1;
	for (size_t i = 1; i < e_s.size(); i++) {
		p *= langmodel[e_s[i]][e_s[i - 1]];
	}

	for (size_t f : f_s) {
		double s = 0;
		for (size_t e : e_s) {
			if (e != e_id2word.size() - 1) s += dict[f][e];
		}
		p *= s;
	}

	return p;
}


void prune(vector<sentence_t>& H, const sentence_t& f_s) {
	while (H.size() > 50) {

		size_t j = 0;
		double m = 9e9;
		for (size_t i = 0; i < H.size(); i++) {
			double s = rate_heuristic(H[i], f_s);
			if (s < m) {
				m = s;
				j = i;
			}
		}
		H.erase(H.begin() + j);
	}
}


void decode_sentence(const sentence_t& f_s) {


	vector<sentence_t> stacks[2] = { { { e_id2word.size() - 1 } }, {} };

	vector<pair<size_t, double>> len;
	for (size_t i = 0; i < e_id2word.size() - 1; i++) len.push_back({i, lenmodel[f_s.size() - 1][i]});
	sort(len.begin(), len.end(), [](const pair<size_t, double>& a, const pair<size_t, double>& b){
		return a.second > b.second;
	});
	size_t l = min(len[0].first + 1, f_s.size() * 4 / 3);


	size_t i;
	for (i = 0; i < l; i++) {

		vector<sentence_t>& H		= stacks[i&1];
		vector<sentence_t>& H_next	= stacks[!(i&1)];
		H_next.clear();


		for (sentence_t& h : H) {

			for (size_t e = 0; e < e_id2word.size() - 1; e++) {

				h.push_back(e);
				H_next.push_back(h);
				h.pop_back();

				prune(H_next, f_s);
			}
		}

		sort(H_next.begin(), H_next.end(), [&](const sentence_t& a, const sentence_t& b) {
			sentence_t aa(a);
			sentence_t bb(b);
			aa.push_back(e_id2word.size() - 1);
			bb.push_back(e_id2word.size() - 1);

			return rate_heuristic(aa, f_s) > rate_heuristic(bb, f_s);
		});

	}

	vector<sentence_t>& H = stacks[i&1];
	for (size_t i = 0; i < min(H.size(), 1ul); i++) {
		H[i].erase(H[i].begin());
		print_sentence(H[i], e_id2word);
	}
}
// </STACK-DECODING>



double rate_sentence(const sentence_t& e_s, const sentence_t& f_s) {

	// language model
	size_t i;
	double p = 1;
	for (i = 1; i < e_s.size(); i++) {
		p *= langmodel[e_s[i]][e_s[i - 1]];
	}
	p *= langmodel[e_s[0]][e_id2word.size()];
	p *= langmodel[e_id2word.size()][e_s[i - 1]];
	p = pow(p, 0.05);

	for (size_t f : f_s) {
		double s = 0;
		for (size_t e : e_s) s += dict[f][e];
		p *= s;
	}

	return p;
}


void hillclimb_sentence(const sentence_t& f_s) {


	// initialise sentence
	sentence_t e_s;
	for (size_t f : f_s) {
		size_t e = 0;
		for (size_t i = 1; i < dict.width(); i++) {
			if (dict[f][i] > dict[f][e]) e = i;
		}
		e_s.push_back(e);
	}
	set<sentence_t> H = { e_s };


	//print_sentence(e_s, e_id2word);


	for (size_t step = 0; step < 1000; step++) {

		auto it = H.begin();
		advance(it, rand() % H.size());
		sentence_t e_s = *it;
		double p = rate_sentence(e_s, f_s);

		bool found;

		switch (rand() % 2) {
		case 0: {	// swap two words
				size_t& e1 = e_s[rand() % e_s.size()];
				size_t& e2 = e_s[rand() % e_s.size()];

				swap(e1, e2);
				double q = rate_sentence(e_s, f_s);
				if (p < q) {
					p = q;
					H.insert(e_s);
					break;
				} else {
					swap(e1, e2);
				}
			}
		case 1: {	// replaceing a word
				found = false;
				size_t& e = e_s[rand() % e_s.size()];
				for (size_t i = 0; i < e_id2word.size(); i++) {
					size_t old_e = e;
					e = i;
					double q = rate_sentence(e_s, f_s);
					if (p < q) {
						found = true;
						p = q;
					} else {
						e = old_e;
					}
				}
				if (found) {
					H.insert(e_s);
					break;
				}
			}
		}


		// prune
		while (H.size() > 10) {
			H.erase(min_element(H.begin(), H.end(), [&](const sentence_t& a, const sentence_t& b){
				return rate_sentence(a, f_s) < rate_sentence(b, f_s);
			}));
		}

	}

	// print all sentences generated
	//for (const sentence_t& e_s : H) print_sentence(e_s, e_id2word);

	// print the best sentence
	e_s = *max_element(H.begin(), H.end(), [&](const sentence_t& a, const sentence_t& b){
		return rate_sentence(a, f_s) < rate_sentence(b, f_s);
	});
	print_sentence(e_s, e_id2word);


	//cout << endl;
}



void decode() {
	load();
	cerr << "Decoding...\n";

//	e_word2id["#"] = e_id2word.size();
//	e_id2word.push_back("#");

	string line;
	while (getline(cin, line)) {
		stringstream words(line);
		sentence_t f_s;

		string word;
		while (words >> word) {
			size_t id;
			if (f_word2id.count(word) == 0) {
				cerr << "Unknown word (" << word << ").\n";
			} else id = f_word2id[word];
			f_s.push_back(id);
		}

		//decode_sentence(f_s);
		hillclimb_sentence(f_s);
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


