#include "utils/helpers.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace cv;
using namespace std;

// ========================= УТИЛИТЫ ПОТОКОВ =========================
void setThreadAffinity(pthread_t thread, int cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    
    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        cerr << "Error setting thread affinity to core " << cpu_core 
             << ": " << strerror(rc) << endl;
    } else {
        cout << "Thread bound to CPU core " << cpu_core << endl;
    }
}

void setCurrentThreadAffinity(int cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        cerr << "Error setting current thread affinity to core " << cpu_core 
             << ": " << strerror(rc) << endl;
    } else {
        cout << "Current thread bound to CPU core " << cpu_core << endl;
    }
}

int getAvailableCores() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

// ========================= УТИЛИТЫ ТОЧЕК =========================
void removeFramePoints(vector<Point2f>& points, double minDistance) {
    if (points.empty()) return;

    sort(points.begin(), points.end(), [](const Point2f& a, const Point2f& b) {
        return a.x < b.x;
    });

    vector<bool> toRemove(points.size(), false);
    for (size_t i = 0; i < points.size(); ++i) {
        if (toRemove[i]) continue;

        for (size_t j = i + 1; j < points.size(); ++j) {
            if (points[j].x - points[i].x > minDistance) break;

            float dx = points[j].x - points[i].x;
            float dy = points[j].y - points[i].y;
            float distanceSq = dx * dx + dy * dy;

            if (distanceSq < minDistance * minDistance) {
                toRemove[j] = true;
            }
        }
    }

    for (int i = (int)points.size() - 1; i >= 0; --i) {
        if (toRemove[i]) {
            points.erase(points.begin() + i);
        }
    }
}
