#pragma once
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <utility>

// ========================= ПОТОКОБЕЗОПАСНАЯ ОЧЕРЕДЬ С ОГРАНИЧЕНИЕМ =========================
template<typename T>
class ThreadSafeQueue {
private:
    deque<T> queue_;
    mutable mutex mutex_;
    condition_variable cond_;
    const size_t maxSize_;
    bool closed_ = false;
    
public:
    explicit ThreadSafeQueue(size_t maxSize = 7) : maxSize_(maxSize) {}
    
    bool push(T data) {
        lock_guard<mutex> lock(mutex_);
        if (closed_ || queue_.size() >= maxSize_) {
            return false;
        }
        queue_.push_back(std::move(data));
        cond_.notify_one();
        return true;
    }
    
    bool try_pop(T& data) {
        lock_guard<mutex> lock(mutex_);
        if (queue_.empty()) return false;
        data = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }
    
    // Блокируется до появления элемента или до shutdown(). Возвращает false после shutdown.
    bool wait_and_pop(T& data) {
        unique_lock<mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return false;
        data = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }
    
    // Разбудить всех ожидающих и запретить дальнейший push (для корректной остановки потоков).
    void shutdown() {
        lock_guard<mutex> lock(mutex_);
        closed_ = true;
        cond_.notify_all();
    }
    
    bool empty() const {
        lock_guard<mutex> lock(mutex_);
        return queue_.empty();
    }
    
    size_t size() const {
        lock_guard<mutex> lock(mutex_);
        return queue_.size();
    }
    
    void clear() {
        lock_guard<mutex> lock(mutex_);
        queue_.clear();
    }
};
