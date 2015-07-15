#pragma once

template <typename T>

struct Vector
{
    T x;
    T y;
    T z;
    
    __device__ T Dot(const Vector <T> B) {
        T ans = x * B.x + y * B.y + z * B.z;
        return ans;
    }
    
    __device__ Vector<T> Cross(const Vector<T> B) {
        Vector<T> ans;
        ans.x = y * B.z - z * B.y;
        ans.y = z * B.x - x * B.z; 
        ans.z = x * B.y - y * B.x;
        return ans;
    }
    
    __device__ Vector<T> operator +(const Vector<T> B) {
        Vector<T> ans;
        ans.x = x + B.x;
        ans.y = y + B.y;
        ans.z = z + B.z;
        return ans;
    }
    
    __device__ Vector<T> operator -(const Vector<T> B) {
        Vector<T> ans;
        ans.x = x - B.x;
        ans.y = y - B.y;
        ans.z = z - B.z;
        return ans;
    }
    
    __device__ Vector<T> operator *(const T B) {
        Vector<T> ans;
        ans.x = x * B;
        ans.y = y * B;
        ans.z = z * B;
        return ans;
    }
    
   __device__ Vector<T> operator /(const T B) {
        Vector<T> ans;
        ans.x = x / B;
        ans.y = y / B;
        ans.z = z / B;
        return ans;
    }
    
    __device__ T Length() {
        T LSq = x * x + y * y + z * z;
        T Ans = sqrt(LSq);
        return Ans;  
    }
    
    __device__ void Normalize() {
        T Len = Length();
        x = x / Len;
        y = y / Len;
        z = z / Len;
    }
    
    
};
