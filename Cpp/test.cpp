#include<iostream>
#include<string>
#include<vector>

using namespace std;

int main(){
  vector<int>v1;
  for(int i=0;i<=10;i++)
    v1.push_back(i);

  for(auto i:v1)
    cout<<i<<endl;
}

