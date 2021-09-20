#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <set>
#include <list>
#include <sstream>
#include <cmath>
#include <queue>
#include <fstream>
#include <string>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <climits>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <map>
#include <deque>

using namespace std;

// INFO(...): output for debugging
ofstream __HEAD_H_FOUT;
#define SIZE(t) (int)(t.size())
#define ALL(t) (t).begin(), (t).end()
#define FOR(i,n) for(int (i)=0; (i)<((int)(n)); (i)++)
#define FORE(i, x) for (__typeof((x).begin()) i = (x).begin(); (i) != (x).end(); (i)++)
// static inline string &ltrim(string &s) { s.erase(s.begin(), find_if( s.begin(), s.end(), not1(ptr_fun<int, int>(isspace)) ) ); return s; }
static inline string &ltrim(string &s) { s.erase(s.begin(), find_if(s.begin(), s.end(), [](int c) {return !std::isspace(c);}));     return s; }
static inline string &rtrim(string &s) { s.erase(find_if(s.rbegin(), s.rend(), [](int c) {return !std::isspace(c);}).base(), s.end()); return s; }
static inline string &trim(string &s) { return ltrim(rtrim(s)); }
string __n_variable(string t, int n){ t=t+','; int i=0; if(n) for(; i<SIZE(t)&&n; i++) if(t[i]==',') n--; n=i; for(;t[i]!=',';i++) continue; t=t.substr(n, i-n); trim(t);if(t[0]=='"') return ""; return t+"="; }
#define __expand_nv(x) __n_variable(t, x)<< t##x << " "
template<class T0>
void ___debug(string t,deque<T0> t0,ostream&os){os<<__n_variable(t,0);FOR(i, SIZE(t0))os<<t0[i]<<" ";}
template<class T0>
void ___debug(string t,set<T0> t0,ostream&os){os<<__n_variable(t,0);FORE(i,t0)os<<*i<<" ";}
template<class T0>
void ___debug(string t,vector<T0> t0,ostream&os){os<<__n_variable(t,0);FOR(i, SIZE(t0))os<<t0[i]<<" ";}
template<class T0,class T1>
void ___debug(string t,vector<pair<T0,T1> > t0,ostream&os){os<<__n_variable(t,0);FOR(i, SIZE(t0))os<<t0[i].F<<","<<t0[i].S<<" ";}
template<class T0>
void ___debug(string t,T0 t0,ostream&os){os<<__expand_nv(0);}
template<class T0,class T1>
void ___debug(string t,T0 t0,T1 t1,ostream&os){os<<__expand_nv(0)<<__expand_nv(1);}
template<class T0,class T1,class T2>
void ___debug(string t,T0 t0,T1 t1,T2 t2,ostream&os){os<<__expand_nv(0)<<__expand_nv(1)<<__expand_nv(2);}
template<class T0,class T1,class T2,class T3>
void ___debug(string t,T0 t0,T1 t1,T2 t2,T3 t3,ostream&os){os<<__expand_nv(0)<<__expand_nv(1)<<__expand_nv(2)<<__expand_nv(3);}
template<class T0,class T1,class T2,class T3,class T4>
void ___debug(string t,T0 t0,T1 t1,T2 t2,T3 t3, T4 t4,ostream&os){os<<__expand_nv(0)<<__expand_nv(1)<<__expand_nv(2)<<__expand_nv(3)<<__expand_nv(4);}

// Commands
#define ASSERT(v) {if (!(v)) {cerr<<"ASSERT FAIL @ "<<__FILE__<<":"<<__LINE__<<endl; exit(1);}}
#define INFO(...) do { ___debug( #__VA_ARGS__,  __VA_ARGS__,cout); cout<<endl; if(__HEAD_H_FOUT.is_open()){___debug( #__VA_ARGS__,  __VA_ARGS__,__HEAD_H_FOUT); __HEAD_H_FOUT<<endl;}  } while(0)