#include <stdio.h> 
#include <unistd.h> 
#include <sys/wait.h> 
#include <sys/types.h> 
#include <stdlib.h>  

  
int main() 
{ 
    int i; 
    int pid = fork(); 
  
    if (pid == 0) { 
        for (i=0; i<15; i++) 
            printf("I am Child #1\n"); 
            while(1);
    } 
    else { 
        printf("I am Parent process, just forked %d\n", pid); 
        int  pid2 = fork();
        if (pid2 == 0) { 
            for (i=0; i<15; i++) 
                printf("I am Child #2\n");
            while(1); 
        } else {
			printf("I am Parent process, just forked %d\n", pid2); 
            exit(0);
        } 
    } 
    return (0);
} 
