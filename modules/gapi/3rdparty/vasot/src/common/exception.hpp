// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation


#ifndef VAS_COMMON_EXCEPTION_HPP
#define VAS_COMMON_EXCEPTION_HPP

#include <vas/common.hpp>

#include <exception>
#include <stdexcept>

#define ETHROW(condition, exception_class, message, ...)                                                               \
    {                                                                                                                  \
        if (!(condition)) {                                                                                            \
            throw std::exception_class(message);                                                                       \
        }                                                                                                              \
    }

#define TRACE(fmt, ...)

#endif // VAS_COMMON_EXCEPTION_HPP
