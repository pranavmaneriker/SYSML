#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>
#include <cstdlib>
#include <unordered_map>
#include <set>
#include <ctime>
#include "parser.hpp"
#include "header.h"
#include "ioFile.h"
#include "strPackage.h"
#include "pbar.h"

using namespace aria::csv;
using namespace std;
using namespace pbar;

#define MAX_STRING 100

int numWalks, walkLength;
char dataDir[MAX_STRING];
int msg_n, subforum_n, user_n, post_n, comment_n, msg_base, subforum_base, user_base;

unordered_map<uint32_t, string> id2user;
unordered_map<uint32_t, string> id2subforum;
vector<uint32_t> postId;
vector<uint32_t> commentId;
unordered_map<uint32_t, uint32_t> msg2user; // post & comment to user
vector<vector<uint32_t>> user2post;
vector<vector<uint32_t>> user2comment;
unordered_map<uint32_t, uint32_t> msg2subforum; // post & comment to subforum
vector<vector<uint32_t>> subforum2post;
unordered_map<uint32_t, uint32_t> comment2post; // comment to post
unordered_map<uint32_t, vector<uint32_t>> post2comment;

int row_n, col_n, col, i;

void getArgs(int argc, char** argv) {
    numWalks = -1;
    walkLength = -1;
    if (argc == 1) {
        cout << "\t-datadir <file>" << endl;
        cout << "\t-numwalks <int>, the number of random walks generated for each node" << endl;
        cout << "\t-walklength <int>, the length of each random walk" << endl;
        INFO("Error: Invalid parameters");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < argc; i++) {
        if (argv[i] == string("-datadir")) {
            strcpy(dataDir, argv[i + 1]);
            if (dataDir[strlen(dataDir) - 1] != '/') {
                strcat(dataDir, "/");
            }
        }
        if (argv[i] == string("-numwalks")) {
            numWalks = atoi(argv[i + 1]);
        }
        if (argv[i] == string("-walklength")) {
            walkLength = atoi(argv[i + 1]);
        }
    }
    INFO("getArgs:\tfinished\n\tArguments received:", dataDir, numWalks, walkLength);
}

void readData() {
    INFO("readData: start");
    // read basic information
    string file_basic = dataDir + (string) "graphInfo.txt";
    ifstream f(file_basic);
    int read_userN, read_subforumN, read_msgN;
    f >> read_msgN >> msg_base >> read_subforumN >> subforum_base >> read_userN >> user_base;
    f.close();
    INFO(msg_base, subforum_base, user_base);
    INFO("\tread graphInfo");
    // read id2subforum
    string file_id2subforum = dataDir + (string) "id2subforum.txt";
    f.open(file_id2subforum);
    CsvParser parser(f);
    parser.delimiter('\t');
    string fields_two[2];
    subforum_n = 0;
    for (auto& row : parser) {
        col = 0;
        for (auto& field : row) {
            fields_two[col++] = field;
        }
        id2subforum[atoi(fields_two[0].c_str())] = fields_two[1];
        subforum_n++;
    }
    ASSERT(subforum_n == read_subforumN);
    subforum2post.resize(subforum_n);
    f.close();
    INFO("\tread id2subforum");

    // read id2user
    string file_id2user = dataDir + (string) "id2user.txt";
    f.open(file_id2user);
    CsvParser parser2(f);
    parser2.delimiter('\t');
    user_n = 0;
    for (auto& row : parser2) {
        col = 0;
        for (auto& field : row) {
            fields_two[col++] = field;
        }
        id2user[atoi(fields_two[0].c_str())] = fields_two[1];
        user_n++;
    }
    ASSERT(user_n == read_userN);
    user2post.resize(user_n);
    user2comment.resize(user_n);
    f.close();
    INFO("\tread id2user");
    
    // read commentAndPostList
    string file_commentAndPostList = dataDir + (string) "commentAndPostList.txt";
    f.open(file_commentAndPostList);
    inputVectorBiPart(&f, &postId, &commentId);
    f.close();
    INFO("\tread commentAndPostList");

    // read comment2post and convert it into post2comment
    string file_comment2post = dataDir + (string) "comment2post.txt";
    f.open(file_comment2post);
    CsvParser parser3(f);
    parser3.delimiter('\t');
    for (auto& row : parser3) {
        col = 0;
        for (auto& field : row) {
            fields_two[col++] = field;
        }
        int comment = atoi(fields_two[0].c_str());
        int post = atoi(fields_two[1].c_str());
        comment2post[comment - msg_base] = post;
        post2comment[post - msg_base].push_back(comment);
    }
    post_n = postId.size();
    comment_n = commentId.size();
    msg_n = post_n + comment_n;
    INFO("\tread comment2post", msg_n, post_n, comment_n);
    ASSERT(msg_n == read_msgN);
    f.close();

    // read msg2subforum and convert it into subforum2post
    string file_msg2subforum = dataDir + (string) "msg2subforum.txt";
    f.open(file_msg2subforum);
    CsvParser parser4(f);
    parser4.delimiter('\t');
    for (auto& row : parser4) {
        col = 0;
        for (auto& field : row) {
            fields_two[col++] = field;
        }
        int msg = atoi(fields_two[0].c_str());
        int subforum = atoi(fields_two[1].c_str());
        msg2subforum[msg - msg_base] = subforum;
        if (find(postId.begin(), postId.end(), msg) != postId.end()) {  // edges only between post and subforum
            subforum2post[subforum - subforum_base].push_back(msg);
        }
    }
    f.close();
    INFO("\tread msg2subforum");

    // read msg2user and convert it into user2msg
    string file_msg2user = dataDir + (string) "msg2user.txt";
    f.open(file_msg2user);
    CsvParser parser5(f);
    parser5.delimiter('\t');
    for (auto& row : parser5) {
        col = 0;
        for (auto& field : row) {
            fields_two[col++] = field;
        }
        int msg = atoi(fields_two[0].c_str());
        int user = atoi(fields_two[1].c_str());
        msg2user[msg - msg_base] = user;
        if (find(postId.begin(), postId.end(), msg) != postId.end()) {
            user2post[user - user_base].push_back(msg);
        }
        else {
            user2comment[user - user_base].push_back(msg);
        }
    }
    f.close();
    INFO("\tread msg2user");
}

void Walk(ofstream *f, uint32_t user) {
    // INFO("Walk:", user);
    uint32_t currentUser = user;
    uint8_t type = 0;
    int currentLength = walkLength;
    uint32_t size = 0;
    uint32_t sizeP = 0;
    uint32_t sizeC = 0;
    uint32_t currentNode = user;
    uint8_t failtime = 0;
    while (currentLength > 0) {
        sizeP = user2post[currentUser - user_base].size();
        sizeC = user2comment[currentUser - user_base].size();
        failtime = 0;
        // INFO(sizeP, sizeC);
        // determine the type
        if (currentLength > 8) {    // length >= 9
            type = (uint8_t) rand() % 7;
        }
        else {
            switch (currentLength)
            {
            case 8:
                type = (uint8_t) rand() % 6 + 1;    // only 1 - 6, since currentLength - length(type 0) = 2
                break;
            case 7:
                type = (uint8_t) rand() % 4 + 3;    // 3 - 6
                break;
            case 6:
                type = (uint8_t) rand() % 3;    // 0, 5 - 6
                if (type > 0) {
                    type += 4;
                }
                break;
            case 5:
                type = (uint8_t) rand() % 2 + 1;    // 1 - 2
                break;
            case 4:
                type = (uint8_t) rand() % 2 + 3;    // 3 - 4
                break;
            case 3:
                type = (uint8_t) rand() % 2 + 5;    // 5 - 6
                break;
            case 2:
                if (sizeP > 0) {
                    size = user2post[currentNode - user_base].size();
                    currentNode = user2post[currentNode - user_base][rand() % size]; // P
                    (*f) << " " << currentNode;
                    currentNode = msg2user[currentNode - msg_base]; // U
                    (*f) << " " << currentNode;
                    currentUser = currentNode;
                    currentLength -= 2;
                }
                else {
                    size = user2comment[currentNode - user_base].size();
                    currentNode = user2comment[currentNode - user_base][rand() % size]; // C
                    (*f) << " " << currentNode;
                    currentNode = msg2user[currentNode - msg_base]; // U
                    (*f) << " " << currentNode;
                    currentUser = currentNode;
                    currentLength -= 2;
                }
                break;
            default:
                ASSERT(currentLength == 1);
                (*f) << " " << currentNode;
                currentLength--;
                break;
            }
        }
        if (currentLength == 0) {
            continue;
        }
        if (sizeC == 0) {
            switch (type)
            {
            case 0:
                type = 6;
                break;
            case 1:
                type = 2;
                break;
            case 4:
                type = 3;
                break;
            case 5:
                type = 6;
                break;
            default:
                break;
            }
        }
        if (sizeP == 0) {
            switch (type)
            {
            case 2:
                type = 1;
                break;
            case 3:
                type = 4;
                break;
            case 6:
                type = 5;
                break;
            default:
                break;
            }
        }
        // extend the walk
        // type = 2;
        // if (currentLength == 80) {
        //     INFO("\t\t", currentUser, currentLength, (int)type);
        // }
        uint32_t tmp_post = 0;
        switch(type) {
            case 0: // UCPSPCU
                size = user2comment[currentNode - user_base].size();
                currentNode = user2comment[currentNode - user_base][rand() % size]; // C
                (*f) << " " << currentNode;
                currentNode = comment2post[currentNode - msg_base]; // P
                (*f) << " " << currentNode;
                currentNode = msg2subforum[currentNode - msg_base]; // S
                (*f) << " " << currentNode;
                size = subforum2post[currentNode - subforum_base].size();
                do{
                    tmp_post = subforum2post[currentNode - subforum_base][rand() % size]; // P
                    failtime++;
                } while (post2comment[tmp_post - msg_base].size() == 0 && failtime <= 10);
                currentNode = tmp_post;
                (*f) << " " << currentNode;
                if (failtime <= 10) {
                    size = post2comment[currentNode - msg_base].size();
                    currentNode = post2comment[currentNode - msg_base][rand() % size]; // C
                    (*f) << " " << currentNode;
                    currentLength--;
                }
                currentNode = msg2user[currentNode - msg_base]; // U
                (*f) << " " << currentNode;
                currentUser = currentNode;
                currentLength -= 5;
                break;
            case 1: // UCPSPU
                size = user2comment[currentNode - user_base].size();
                currentNode = user2comment[currentNode - user_base][rand() % size]; // C
                (*f) << " " << currentNode;
                currentNode = comment2post[currentNode - msg_base]; // P
                (*f) << " " << currentNode;
                currentNode = msg2subforum[currentNode - msg_base]; // S
                (*f) << " " << currentNode;
                size = subforum2post[currentNode - subforum_base].size();
                currentNode = subforum2post[currentNode - subforum_base][rand() % size]; // P
                (*f) << " " << currentNode;
                currentNode = msg2user[currentNode - msg_base]; // U
                (*f) << " " << currentNode;
                currentUser = currentNode;
                currentLength -= 5;
                break;
            case 2: // UPSPCU
                size = user2post[currentNode - user_base].size();
                currentNode = user2post[currentNode - user_base][rand() % size]; // P
                (*f) << " " << currentNode;
                currentNode = msg2subforum[currentNode - msg_base]; // S
                (*f) << " " << currentNode;
                size = subforum2post[currentNode - subforum_base].size();
                do{
                    tmp_post = subforum2post[currentNode - subforum_base][rand() % size]; // P
                    failtime++;
                } while (post2comment[tmp_post - msg_base].size() == 0 && failtime <= 10);
                currentNode = tmp_post;
                (*f) << " " << currentNode;
                if (failtime <= 10) {   // if no possible SPCU instance, use SPU instead
                    size = post2comment[currentNode - msg_base].size();
                    currentNode = post2comment[currentNode - msg_base][rand() % size]; // C
                    (*f) << " " << currentNode;
                    currentLength--;
                }
                currentNode = msg2user[currentNode - msg_base]; // U
                (*f) << " " << currentNode;
                currentUser = currentNode;
                currentLength -= 4;
                break;
            case 3: // UPSPU
                size = user2post[currentNode - user_base].size();
                currentNode = user2post[currentNode - user_base][rand() % size]; // P
                (*f) << " " << currentNode;
                currentNode = msg2subforum[currentNode - msg_base]; // S
                (*f) << " " << currentNode;
                size = subforum2post[currentNode - subforum_base].size();
                currentNode = subforum2post[currentNode - subforum_base][rand() % size]; // P
                (*f) << " " << currentNode;
                currentNode = msg2user[currentNode - msg_base]; // U
                (*f) << " " << currentNode;
                currentUser = currentNode;
                currentLength -= 4;
                break;
            case 4: // UCPCU
                size = user2comment[currentNode - user_base].size();
                currentNode = user2comment[currentNode - user_base][rand() % size]; // C
                (*f) << " " << currentNode;
                currentNode = comment2post[currentNode - msg_base]; // P
                (*f) << " " << currentNode;
                size = post2comment[currentNode - msg_base].size();
                currentNode = post2comment[currentNode - msg_base][rand() % size]; // C
                (*f) << " " << currentNode;
                currentNode = msg2user[currentNode - msg_base]; // U
                (*f) << " " << currentNode;
                currentUser = currentNode;
                currentLength -= 4;
                break;
            case 5: // UCPU
                size = user2comment[currentNode - user_base].size();
                currentNode = user2comment[currentNode - user_base][rand() % size]; // C
                (*f) << " " << currentNode;
                currentNode = comment2post[currentNode - msg_base]; // P
                (*f) << " " << currentNode;
                currentNode = msg2user[currentNode - msg_base]; // U
                (*f) << " " << currentNode;
                currentUser = currentNode;
                currentLength -= 3;
                break;
            default: // UPCU
                size = user2post[currentNode - user_base].size();
                do{
                    tmp_post = user2post[currentNode - user_base][rand() % size]; // P
                    failtime++;
                } while (post2comment[tmp_post - msg_base].size() == 0 && failtime <= 10);
                currentNode = tmp_post;
                (*f) << " " << currentNode;
                if (failtime <= 10) {
                    size = post2comment[currentNode - msg_base].size();
                    currentNode = post2comment[currentNode - msg_base][rand() % size]; // C
                    (*f) << " " << currentNode;
                    currentLength--;
                }
                currentNode = msg2user[currentNode - msg_base]; // U
                (*f) << " " << currentNode;
                currentUser = currentNode;
                currentLength -= 2;
                break;
        }
    }
    (*f) << endl;
}

void generateWalks() {
    INFO("generateWalks: start");
    // check if dataDir exists, mkdir if not
    if (!isFolderExist(dataDir)) {
        int dummy = createDirectory(dataDir);
        INFO("generateWalks:\tdirectory", dataDir ,"created");
    }
    // create ofstream
    string identDir = slashToUnderscore(((string)dataDir).substr(0, strlen(dataDir) - 2));
    string outputFile = dataDir + (string) "walks_" + identDir + (string) "_" + to_string(numWalks) + (string) "_" + to_string(walkLength) + (string) ".txt";
    ofstream fout(outputFile);
    fout << user_n << " " << numWalks << " " << walkLength << endl;
    srand(time(NULL)); // set random seed
    // generate walks
    uint32_t j;
    ProgressBar<unordered_map<uint32_t, string>::iterator> pbar(id2user.begin(), id2user.end(), 50);
    uint32_t userId = 0;
    for (auto& user : pbar) {
        for (j = 0; j < numWalks; j++) {
            // INFO(user.first, user.second, j);
            fout << user.first;
            Walk(&fout, user.first);
        }
    }
    // close f
    INFO("generateWalks:\tfinished");
}

int main(int argc, char** argv) {
    getArgs(argc, argv); // get arguments
    readData();
    generateWalks();
    return 0;
}