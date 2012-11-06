#include <signal.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace std;

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
		fill(0);
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
			} else {
				id = word2id[word];
			}
			sentence.push_back(id);
		}
		maxlen = max(maxlen, sentence.size());
		corpus.push_back(sentence);
	}
	return maxlen;
}


void normalize(matrix<float>& m) {
	for (size_t x = 0; x < m.width(); x++) {
		float s = 0;
		for (size_t y = 0; y < m.height(); y++) s += m[y][x];
		for (size_t y = 0; y < m.height(); y++) m[y][x] /= s;
	}
}


string					base;
string					e_lang;
string					f_lang;
map<string, size_t>		e_word2id;
map<string, size_t>		f_word2id;
vector<string>			e_id2word;
vector<string>			f_id2word;
matrix<float>			dict;


void top_ten(const string word) {
	cout << word << "\n";
	if (!f_word2id.count(word)) {
		cout << "\tUnknown word.\n";
		return;
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


void save() {
	cout << "Saving...\n";
	string filename = base + ".meta-" + f_lang + "-to-" + e_lang;
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "Error opening file " << filename << ".\n";
	}

	file << f_id2word.size() << "\n";
	for (auto s : f_id2word) file << s << "\n";
	file << e_id2word.size() << "\n";
	for (auto s : e_id2word) file << s << "\n";

	file.write((char*)dict.ptr(), sizeof(float) * f_id2word.size() * e_id2word.size());
}


void load() {
	cout << "Loading...\n";
	string filename = base + ".meta-" + f_lang + "-to-" + e_lang;
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error opening file " << filename << ".\n";
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
	file.read((char*)dict.ptr(), sizeof(float) * f_id2word.size() * e_id2word.size());

}


void leaving(int sig) {
	save();
	exit(0);
}


void train(int iterations) {
	cout << "Reading corpora...\n";
	vector<vector<size_t>>	e_corpus;
	vector<vector<size_t>>	f_corpus;
	size_t f_msl = read_corpus(base + "." + f_lang, f_word2id, f_id2word, f_corpus);
	size_t e_msl = read_corpus(base + "." + e_lang, e_word2id, e_id2word, e_corpus);
	if (e_corpus.size() != f_corpus.size()) {
		cerr << "Corpora size differs.\n";
		exit(1);
	}


	// language model (not used as yet)
	cout << "Generating language model...\n";
	matrix<float> langmodel;
	langmodel.init(e_id2word.size() + 1, e_id2word.size() + 1);
	for (auto sentence : e_corpus) {
		size_t prev_id = e_id2word.size();
		for (size_t id : sentence) {
			langmodel[id][prev_id]++;
			prev_id = id;
		}
		langmodel[e_id2word.size()][prev_id]++;
	}
	normalize(langmodel);


	// length model (not used as yet)
	cout << "Generating length model...\n";
	matrix<float> lenmodel;
	lenmodel.init(f_msl, e_msl);
	for (size_t l = 0; l < e_corpus.size(); l++) {
		lenmodel[f_corpus[l].size() - 1][e_corpus[l].size() - 1]++;
	}
	normalize(lenmodel);



	// dictionary training
	matrix<float> c;
	dict.init(f_id2word.size(), e_id2word.size());
	c.init(f_id2word.size(), e_id2word.size());
	dict.fill(1);

	signal(SIGINT, leaving);
	cout << "Training dictionary...\n";
	for (int count = 0; count < iterations; count++) {
		cout << "Step " << count + 1 << "...\n";

		c.fill(0);
		for (size_t l = 0; l < e_corpus.size(); l++) {
			for (size_t f : f_corpus[l]) {
				float s = 0;
				for (size_t e : e_corpus[l]) s += dict[f][e];
				s = 1 / s;
				for (size_t e : e_corpus[l]) c[f][e] += dict[f][e] * s;
			}
		}
		normalize(c);
		dict.swap(c);
	}
	save();
}


void lookup() {
	load();
	cout << "Reading input...\n";
	string s;
	while (getline(cin, s)) top_ten(s);
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
	else {
		cerr << "Invalid action.\n";
		return 1;
	}
	return 0;
}

