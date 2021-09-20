#include <iostream>
#include <cstring>
#include <ctime>

using namespace std;

int strtotime(string datetime)  
{  
    struct tm tm_time;  
    int unixtime;  
    strptime(datetime.c_str(), "%Y-%m-%d %H:%M:%S", &tm_time);  
    
    unixtime = mktime(&tm_time);  
    return unixtime;  
}

int strtotime2(string datetime)  
{  
    struct tm tm_time;  
    int unixtime;  
    strptime(datetime.c_str(), "%B %d, %Y, %H:%M:%S", &tm_time);  
    
    unixtime = mktime(&tm_time);  
    return unixtime;  
}

int strtotime3(string datetime)  
{  
    struct tm tm_time;  
    int unixtime;  
    strptime(datetime.c_str(), "%Y-%m-%d %H:%M", &tm_time);  
    
    unixtime = mktime(&tm_time);  
    return unixtime;  
}

string slashToUnderscore(string dir)
{
    replace(dir.begin(), dir.end(), '/', '_');
    return dir;
}