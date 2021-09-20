#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <map>
#include <unordered_map>
#include <iterator>
#include <cstring>
#include <unistd.h> // _access func
#include <sys/types.h> // _mkdir func
#include <sys/stat.h> // _mkdir func
#include <dirent.h>

using namespace std;

template<typename T>
void outputVector_wId(ofstream *f, vector<T> v, int base, char seg) {
    if (!(*f).is_open())
        exit(EXIT_FAILURE);
    int size = v.size();
    for (int i = 0; i < size; i++) {
        (*f) << (base + i) << seg << v[i] << endl;
    }
}

template<typename T>
void outputVecVec_wId(ofstream *f, vector<vector<T>> v, int base, char seg) {
    if (!(*f).is_open())
        exit(EXIT_FAILURE);
	int size = v.size();
	(*f) << size << endl;
	int i, j;
	for (i = 0; i < size; i++) {
		int size_vec = v[i].size();
		(*f) << size_vec;
		for (j = 0; j < size_vec; j++) {
			(*f) << " " << v[i][j];
		}
		(*f) << endl;
	}
	(*f).close();
}

template<typename T1, typename T2>
void outputMap(ofstream *f, map<T1, T2> m, char seg) {
    if (!(*f).is_open())
        exit(EXIT_FAILURE);
	typename map<T1, T2>::iterator iter;
	iter = m.begin();
    while (iter != m.end()) {
        (*f) << iter->first << seg << iter->second << endl;
		iter++;
    }
	(*f).close();
}

template<typename T1, typename T2>
void outputUmap(ofstream *f, unordered_map<T1, T2> m, char seg) {
    if (!(*f).is_open())
        exit(EXIT_FAILURE);
	typename unordered_map<T1, T2>::iterator iter;
	iter = m.begin();
    while (iter != m.end()) {
        (*f) << iter->first << seg << iter->second << endl;
		iter++;
    }
	(*f).close();
}

// template<typename T1, typename T2>
// void outputUmapBiPart(ofstream *f, unordered_map<T1, T2> m, int offset, int n, char seg) {
// 	if (!(*f).is_open())
//         exit(EXIT_FAILURE);
// 	typename unordered_map<T1, T2>::iterator iter;
// 	iter = m.begin();
// 	while (iter != m.end()) {
//         (*f) << iter->first << seg;
// 		iter++;
//     }
// 	(*f) << endl;
// 	for (int i = 0; i < n; i++) {
// 		if (m.find(offset + i) == m.end()) {
// 			(*f) << offset + i << seg;
// 		}
// 	}
// 	(*f) << endl;
// 	(*f).close();
// }

template<typename T1, typename T2>
void inputVectorBiPart(ifstream *f, vector<T1> *a, vector<T2> *b) {
	if (!(*f).is_open())
        exit(EXIT_FAILURE);
	int n, offset;
	(*f) >> n >> offset;
	char c;
	for (int i = offset; i < offset + n; i++) {
		(*f) >> c;
		if (c == '0') {
			(*a).push_back(i);
		}
		else {
			(*b).push_back(i);
		}
	}
	(*f).close();
}

template<typename T1, typename T2>
void outputUmapBiPart(ofstream *f, unordered_map<T1, T2> m, int offset, int n) {
	if (!(*f).is_open())
        exit(EXIT_FAILURE);
	(*f) << n << " " << offset << endl;
	for (int i = 0; i < n; i++) {
		if (m.find(offset + i) == m.end()) {
			(*f) << '0';
		}
		else {
			(*f) << '1';
		}
	}
	(*f) << endl;
	(*f).close();
}

bool isFolderExist(char * folder)
{
	int ret = 0;
 
	ret = access(folder, 0);
	if (ret == 0)
		ret = true;
	else
		ret = false;
 
	return ret;
}

vector<string> getFileswithExt(const char* path, const char* prefix, const char* subfix) {
	vector<string> files;

	struct dirent *entry;
	DIR *dir = opendir(path);
	int lengPrefix = strlen(prefix);
	int lengSubfix = strlen(subfix);

	if (dir == NULL) {
	  return files;
	}
	while ((entry = readdir(dir)) != NULL) {
		string filename = entry->d_name;
		if (filename.substr(0, lengPrefix) == string(prefix) && filename.substr(filename.length() - lengSubfix) == string(subfix))
			files.push_back(filename);
	}
	closedir(dir);
	return files;
}

void mergeCSV(const char* path, const char* prefix, const char* subfix) {
	// merge eligible csv files with headers removed
	INFO("mergeCSV: ", path, prefix, subfix);
	string command1 = "cd " + (string) path;
    string command2 = "tail -qn +2 " + (string) prefix + "*" + (string) subfix + " > merged.csv2";
	string command3 = "mv merged.csv2 merged.csv";
	string command = command1 + "; " + command2 + "; " + command3;
	system(command.c_str());
	// add header line
	vector<string> files = getFileswithExt(path, prefix, subfix);
	INFO("mergeCSV:", files.size());
	ifstream f((string) path + files.at(0));
	string line;
	getline(f, line);
	INFO("mergeCSV: Read header", line);
	// read first line and add
	command1 = "cd " + (string) path;
	INFO("mergeCSV: " + command1);
	command2 = "echo \"" + line + "\\n$(cat merged.csv)\" > merged.csv";
	INFO("mergeCSV: " + command2);
	command = command1 + "; " + command2;
	system(command.c_str());
}

int32_t createDirectory(char* directoryPath)
{
	uint32_t dirPathLen = 0;
	if (directoryPath != NULL) {
		dirPathLen = strlen(directoryPath);
	}
	if (directoryPath[dirPathLen - 1] != '/') {
		strcat(directoryPath, "/");
		dirPathLen = strlen(directoryPath);
	}
	if (dirPathLen > FILENAME_MAX)
	{
		return -1;
	}
	char tmpDirPath[FILENAME_MAX] = { 0 };
	for (uint32_t i = 0; i < dirPathLen; ++i)
	{
		tmpDirPath[i] = directoryPath[i];
		if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/')
		{
			if (!isFolderExist(tmpDirPath))
			{
				int ret = mkdir(tmpDirPath, ACCESSPERMS);
				//BOOL ret = CreateDirectory(tmpDirPath, NULL);
				if (ret != 0)
					return -1;
			}
		}
	}
	return 0;
}