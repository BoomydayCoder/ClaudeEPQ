// Process start-up benchmark: spawnbench N STDIN_FILE CMD [ARGS...]
// Runs CMD N times (stdin from STDIN_FILE, stdout/stderr to /dev/null) and
// prints min / median / mean wall time per run in microseconds.
#include <fcntl.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char** environ;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
static int cmp(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: spawnbench N STDIN_FILE CMD [ARGS...]\n");
        return 2;
    }
    int n = atoi(argv[1]);
    double* t = malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) {
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addopen(&fa, 0, argv[2], O_RDONLY, 0);
        posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
        posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
        double t0 = now_us();
        pid_t pid;
        if (posix_spawn(&pid, argv[3], &fa, NULL, argv + 3, environ) != 0) {
            perror("spawn");
            return 1;
        }
        int st;
        waitpid(pid, &st, 0);
        t[i] = now_us() - t0;
        posix_spawn_file_actions_destroy(&fa);
    }
    qsort(t, (size_t)n, sizeof(double), cmp);
    double s = 0;
    for (int i = 0; i < n; i++) s += t[i];
    printf("min %.0f us  median %.0f us  mean %.0f us  (n=%d)\n", t[0], t[n / 2], s / n, n);
    return 0;
}
