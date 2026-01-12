#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <stdio.h>

struct Timer {
    struct timeval start, end;
};

void startTimer(struct Timer* t) {
    gettimeofday(&t->start, NULL);
}

double stopTimer(struct Timer* t) {
    gettimeofday(&t->end, NULL);
    return (t->end.tv_sec - t->start.tv_sec) + 
           (t->end.tv_usec - t->start.tv_usec) / 1e6;
}

void printElapsedTime(struct Timer* t, const char* name) {
    double elapsed = stopTimer(t);
    printf("%s: %f seconds\n", name, elapsed);
}

#endif