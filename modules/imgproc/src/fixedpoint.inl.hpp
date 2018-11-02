// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#ifndef _CV_FIXEDPOINT_HPP_
#define _CV_FIXEDPOINT_HPP_

#include "opencv2/core/softfloat.hpp"

#ifndef CV_ALWAYS_INLINE
    #if defined(__GNUC__) && (__GNUC__ > 3 ||(__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
        #define CV_ALWAYS_INLINE inline __attribute__((always_inline))
    #elif defined(_MSC_VER)
        #define CV_ALWAYS_INLINE __forceinline
    #else
        #define CV_ALWAYS_INLINE inline
    #endif
#endif

namespace
{

class fixedpoint64
{
public:
    static const int fixedShift = 32;

    int64_t val;
    fixedpoint64(int64_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint64_t fixedround(const uint64_t& _val) { return (_val + ((1LL << fixedShift) >> 1)); }
    
    typedef fixedpoint64 WT;
    CV_ALWAYS_INLINE fixedpoint64() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint64(const cv::softdouble& _val) { val = cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); }
    CV_ALWAYS_INLINE fixedpoint64 operator >> (int n) const { return fixedpoint64(val >> n); }
    CV_ALWAYS_INLINE fixedpoint64 operator << (int n) const { return fixedpoint64(val << n); }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>((int64_t)fixedround((uint64_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint64 zero() { return fixedpoint64(); }
    static CV_ALWAYS_INLINE fixedpoint64 one() { return fixedpoint64((int64_t)(1LL << fixedShift)); }
    friend class fixedpoint32;
};

class ufixedpoint64
{
public:
    static const int fixedShift = 32;

    uint64_t val;
    ufixedpoint64(uint64_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint64_t fixedround(const uint64_t& _val) { return (_val + ((1LL << fixedShift) >> 1)); }
    
    typedef ufixedpoint64 WT;
    CV_ALWAYS_INLINE ufixedpoint64() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint64(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint64_t)cvRound64(_val * cv::softdouble((int64_t)(1LL << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint64 operator >> (int n) const { return ufixedpoint64(val >> n); }
    CV_ALWAYS_INLINE ufixedpoint64 operator << (int n) const { return ufixedpoint64(val << n); }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1LL << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint64 zero() { return ufixedpoint64(); }
    static CV_ALWAYS_INLINE ufixedpoint64 one() { return ufixedpoint64((uint64_t)(1ULL << fixedShift)); }
    friend class ufixedpoint32;
};

class fixedpoint32
{
public:
    static const int fixedShift = 16;

    int32_t val;
    fixedpoint32(int32_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint32_t fixedround(const uint32_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
    
    typedef fixedpoint64 WT;
    CV_ALWAYS_INLINE fixedpoint32() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint32(const cv::softdouble& _val) { val = (int32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    CV_ALWAYS_INLINE fixedpoint32 operator >> (int n) const { return fixedpoint32(val >> n); }
    CV_ALWAYS_INLINE fixedpoint32 operator << (int n) const { return fixedpoint32(val << n); }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>((int32_t)fixedround((uint32_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator fixedpoint64() const { return (int64_t)val << (fixedpoint64::fixedShift - fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint32 zero() { return fixedpoint32(); }
    static CV_ALWAYS_INLINE fixedpoint32 one() { return fixedpoint32((1 << fixedShift)); }
    friend class fixedpoint16;
};

class ufixedpoint32
{
public:
    static const int fixedShift = 16;

    uint32_t val;
    ufixedpoint32(uint32_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint32_t fixedround(const uint32_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
    
    typedef ufixedpoint64 WT;
    CV_ALWAYS_INLINE ufixedpoint32() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint32(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint32_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint32 operator >> (int n) const { return ufixedpoint32(val >> n); }
    CV_ALWAYS_INLINE ufixedpoint32 operator << (int n) const { return ufixedpoint32(val << n); }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator ufixedpoint64() const { return (uint64_t)val << (ufixedpoint64::fixedShift - fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint32 zero() { return ufixedpoint32(); }
    static CV_ALWAYS_INLINE ufixedpoint32 one() { return ufixedpoint32((1U << fixedShift)); }
    friend class ufixedpoint16;
};

class fixedpoint16
{
public:
    static const int fixedShift = 8;

    int16_t val;
    fixedpoint16(int16_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint16_t fixedround(const uint16_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
    
    typedef fixedpoint32 WT;
    CV_ALWAYS_INLINE fixedpoint16() { val = 0; }
    CV_ALWAYS_INLINE fixedpoint16(const cv::softdouble& _val) { val = (int16_t)cvRound(_val * cv::softdouble((1 << fixedShift))); }
    //CV_ALWAYS_INLINE fixedpoint16& operator = (const int8_t& _val) { val = ((int16_t)_val) << fixedShift; return *this; }
    //CV_ALWAYS_INLINE fixedpoint16& operator = (const cv::softdouble& _val) { val = (int16_t)cvRound(_val * cv::softdouble((1 << fixedShift))); return *this; }
    //CV_ALWAYS_INLINE fixedpoint16& operator = (const fixedpoint16& _val) { val = _val.val; return *this; }
    CV_ALWAYS_INLINE fixedpoint16 operator >> (int n) const { return fixedpoint16((int16_t)(val >> n)); }
    CV_ALWAYS_INLINE fixedpoint16 operator << (int n) const { return fixedpoint16((int16_t)(val << n)); }
    //CV_ALWAYS_INLINE bool operator == (const fixedpoint16& val2) const { return val == val2.val; }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>((int16_t)fixedround((uint16_t)val) >> fixedShift); }
    CV_ALWAYS_INLINE operator fixedpoint32() const { return (int32_t)val << (fixedpoint32::fixedShift - fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE fixedpoint16 zero() { return fixedpoint16(); }
    static CV_ALWAYS_INLINE fixedpoint16 one() { return fixedpoint16((int16_t)(1 << fixedShift)); }
};

class ufixedpoint16
{
public:
    static const int fixedShift = 8;

    uint16_t val;
    ufixedpoint16(uint16_t _val) : val(_val) {}
    static CV_ALWAYS_INLINE uint16_t fixedround(const uint16_t& _val) { return (_val + ((1 << fixedShift) >> 1)); }
    
    typedef ufixedpoint32 WT;
    CV_ALWAYS_INLINE ufixedpoint16() { val = 0; }
    CV_ALWAYS_INLINE ufixedpoint16(const cv::softdouble& _val) { val = _val.getSign() ? 0 : (uint16_t)cvRound(_val * cv::softdouble((int32_t)(1 << fixedShift))); }
    CV_ALWAYS_INLINE ufixedpoint16 operator >> (int n) const { return ufixedpoint16((uint16_t)(val >> n)); }
    CV_ALWAYS_INLINE ufixedpoint16 operator << (int n) const { return ufixedpoint16((uint16_t)(val << n)); }
    template <typename ET>
    CV_ALWAYS_INLINE ET saturate_cast() const { return cv::saturate_cast<ET>(fixedround(val) >> fixedShift); }
    CV_ALWAYS_INLINE operator ufixedpoint32() const { return (uint32_t)val << (ufixedpoint32::fixedShift - fixedShift); }
    CV_ALWAYS_INLINE operator float() const { return (float)val / (1 << fixedShift); }
    CV_ALWAYS_INLINE bool isZero() { return val == 0; }
    static CV_ALWAYS_INLINE ufixedpoint16 zero() { return ufixedpoint16(); }
    static CV_ALWAYS_INLINE ufixedpoint16 one() { return ufixedpoint16((uint16_t)(1 << fixedShift)); }
};

}

#endif
