#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cstdlib>
#include <unordered_map>
#include <ctime>
#include "parser.hpp"
#include "header.h"
#include "ioFile.h"
#include "strPackage.h"

using namespace aria::csv;
using namespace std;

#define MAX_STRING 100

int padding = -1;
char raw_data_file[MAX_STRING], outputDir[MAX_STRING];
int msg_base, msg_n;
int subforum_base, subforum_n;
int user_base, user_n;
bool msglist = false;

unordered_map<uint32_t, uint32_t> msg2user; // post & comment to user
unordered_map<uint32_t, uint32_t> msg2subforum; // post & comment to subforum
unordered_map<uint32_t, uint32_t> comment2post; // comment to post
vector<string> id2user;
vector<string> id2subforum;
vector<string> id2msg;

int row_n, col_n, col;

void getArgs(int argc, char** argv) {
    padding = -1;
    if (argc == 1) {
        cout << "\t-rawdata <file>" << endl;
        cout << "\t-outputdir <dir>" << endl;
        cout << "\t-padding <int>, the gap between node groups of different types" << endl;
        cout << "\t-msglist, whether outputs the post and comment or not" << endl;
        INFO("Error: Invalid parameters");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < argc; i++) {
        if (argv[i] == string("-rawdata")) {
            strcpy(raw_data_file, argv[i + 1]);
        }
        if (argv[i] == string("-outputdir")) {
            strcpy(outputDir, argv[i + 1]);
            if (outputDir[strlen(outputDir) - 1] != '/') {
                strcat(outputDir, "/");
            }
        }
        if (argv[i] == string("-padding")) {
            padding = atoi(argv[i + 1]);
        }
        if (argv[i] == string("-msglist")) {
            msglist = true;
        }
    }
    INFO("getArgs:\tfinished\n\tArguments received:", raw_data_file, outputDir, padding, msglist);
}

void initialize() {
    if (padding == -1)
        padding = 3000000;
    msg_base = 0;
    msg_n = 0;
    subforum_base = msg_base + padding;
    subforum_n = 0;
    user_base = subforum_base + padding;
    user_n = 0;
    INFO("initialize:\tfinished");
}

void output() {
    // check if outputDir exists, mkdir if not
    if (!isFolderExist(outputDir)) {
        int dummy = createDirectory(outputDir);
        INFO("output:\tdirectory", outputDir ,"created");
    }
    // output basic information
    string outputFile_basic = outputDir + (string) "graphInfo.txt";
    ofstream fout(outputFile_basic);
    fout << msg_n << " " << msg_base << " " << subforum_n << " " << subforum_base << " " << user_n << " " << user_base << endl;
    fout.close();
    // output id of username, subforum, post & comment
    char segment = '\t';
    string outputFile_id2subforum = outputDir + (string) "id2subforum.txt";
    fout.open(outputFile_id2subforum);
    outputVector_wId(&fout, id2subforum, subforum_base, segment);
    fout.close();
    string outputFile_id2user = outputDir + (string) "id2user.txt";
    fout.open(outputFile_id2user);
    outputVector_wId(&fout, id2user, user_base, segment);
    fout.close();
    if (msglist) {
        string outputFile_id2msg = outputDir + (string) "id2msg.txt";
        fout.open(outputFile_id2msg);
        outputVector_wId(&fout, id2msg, msg_base, segment);
        fout.close();
    }
    // output links of msg2user, msg2subforum, comment2post
    string outputFile_msg2user = outputDir + (string) "msg2user.txt";
    fout.open(outputFile_msg2user);
    outputUmap(&fout, msg2user, segment);
    fout.close();
    string outputFile_msg2subforum = outputDir + (string) "msg2subforum.txt";
    fout.open(outputFile_msg2subforum);
    outputUmap(&fout, msg2subforum, segment);
    fout.close();
    string outputFile_comment2post = outputDir + (string) "comment2post.txt";
    fout.open(outputFile_comment2post);
    outputUmap(&fout, comment2post, segment);
    fout.close();
    string outputFile_commentAndPostList = outputDir + (string) "commentAndPostList.txt";
    fout.open(outputFile_commentAndPostList);
    outputUmapBiPart(&fout, comment2post, msg_base, msg_n);
    // close f
    INFO("output:\tfinished");
}

void readNetwork() {
    // initialize
    ifstream f(raw_data_file);
    CsvParser parser(f);
    // parser.delimiter(',');
    int i;
    unordered_map<string, uint32_t> subforum2id;
    unordered_map<string, uint32_t> user2id;
    unordered_map<string, uint32_t> title2earliestTime;
    unordered_map<string, uint32_t> title2post;
    vector<string> titles;

    row_n = 0, col_n = 6, col = 0;
    msg_n = -2;

    const int col_date = 3;
    const int col_title = 4;
    const int col_subforum = 1;
    const int col_username = 2;
    const int col_msg = 5;

    string fields[col_n];
    for (auto& row : parser) {
        // initialize
        row_n++;
        msg_n++;
        if (msg_n == -1) continue;
        col = 0;

        // copy each field in the row
        for (auto& field : row) {
            fields[col++] = field;
        }

        // assign id to new subforum / post & comment / username
        for (col = 0; col < col_n; col++) {
            string field = fields[col];
            switch (col)
            {
            case col_subforum:     // subforum
                if (subforum2id.find("subforum_" + field) == subforum2id.end()) {
                    string subforum = "subforum_" + field;
                    subforum2id[subforum] = subforum_base + subforum_n;
                    id2subforum.push_back(subforum);
                    subforum_n++;
                }
                
                break;
            
            case col_msg:     // post & comment
                if (msglist) {
                    id2msg.push_back(field);
                }
                break;
            
            case col_username:     // username
                if (user2id.find("user_" + field) == user2id.end()) {
                    string username = "user_" + field;
                    user2id[username] = user_base + user_n;
                    id2user.push_back(username);
                    user_n++;
                }
                break;
            
            case col_title:
                titles.push_back(field);
                ASSERT(titles.size() == msg_n + 1);
                break;

            default:
                break;
            }
        }

        // get post time and compare, choose the earliest one as post and others as its comments
        if (title2earliestTime.find(fields[col_title]) == title2earliestTime.end() || strtotime3(fields[col_date]) < title2earliestTime[fields[col_title]]) {
            title2earliestTime[fields[col_title]] = strtotime3(fields[col_date]);
            title2post[fields[col_title]] = msg_base + msg_n;
        }
                
        // add mappings among post & comment, user, subforum
        msg2user[(uint32_t) (msg_base + msg_n)] = user2id["user_" + fields[col_username]];
        msg2subforum[(uint32_t) (msg_base + msg_n)] = subforum2id["subforum_" + fields[col_subforum]];
    }
    msg_n++;
    ASSERT(row_n == msg_n + 1);

    // add mappings from comment to post
    for (i = 0; i < msg_n; i++) {
        if (title2post[titles[i]] != msg_base + i) {
            comment2post[uint32_t (msg_base + i)] = title2post[titles[i]];
        }
    }
    unordered_map<string, uint32_t>().swap(title2earliestTime);
    unordered_map<string, uint32_t>().swap(title2post);
    vector<string>().swap(titles);

    // validation
    for (i = 0; i < subforum_n; i++) {
        ASSERT(subforum2id[id2subforum[i]] == subforum_base + i);
    }
    for (i = 0; i < user_n; i++) {
        ASSERT(user2id[id2user[i]] == user_base + i);
    }
    unordered_map<string, uint32_t>().swap(subforum2id);
    unordered_map<string, uint32_t>().swap(user2id);

    // show statistics
    INFO("readNetwork:\tfinished\n\tStatistics:",msg_n, subforum_n, user_n);
}

void addDummySubforum() {
    string newSubforum = "subforum_testnew";
    id2subforum.push_back(newSubforum);
    uint32_t newSubforum_id = subforum_base + subforum_n;
    subforum_n++;
    unordered_map<uint32_t, uint32_t> ori;
    for (auto &myPair : msg2subforum) {
        if (myPair.second == newSubforum_id - 1) {
            ori[myPair.first] = msg_base + msg_n;
            msg2subforum[msg_base + msg_n] = newSubforum_id;
            msg2user[msg_base + msg_n] = msg2user[myPair.first];
            msg_n++;
        }
    }
    for (auto &c2p : comment2post) {
        if (msg2subforum[c2p.first] == newSubforum_id - 1) {
            comment2post[ori[c2p.first]] = ori[c2p.second];
        }
    }
}

int main(int argc, char** argv) {
    getArgs(argc, argv); // get arguments
    initialize();
    readNetwork();
    // addDummySubforum();
    output();
    return 0;
}
