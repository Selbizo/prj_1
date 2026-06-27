#include "utils/helpers.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace cv;
using namespace std;

// ========================= УТИЛИТЫ ПУТИ =========================
string normalizePath(const string& path) {
    string p = path;
    if (!p.empty() && p.back() != '/') {
        p += '/';
    }
    return p;
}

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

// ========================= УТИЛИТЫ КАРТИНКИ =========================
void loadImage(Mat& image, int frame_id, const string& filepath) {
    char file[200];
    
    sprintf(file, "image_0/%06d.png", frame_id);
    string filename = filepath + file;
    
    ifstream file_check(filename.c_str());
    if (file_check.good()) {
        file_check.close();
        image = imread(filename, IMREAD_COLOR);
        if (!image.empty()) return;
    }
    
    sprintf(file, "image_0/%06d.jpg", frame_id);
    filename = filepath + file;
    
    ifstream jpg_check(filename.c_str());
    if (jpg_check.good()) {
        jpg_check.close();
        image = imread(filename, IMREAD_COLOR);
    }
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
