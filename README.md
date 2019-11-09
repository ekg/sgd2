# header-only s_gd2

This is a simple refactoring of the C++ code from https://github.com/jxz12/s_gd2 into a header-only library.

It updates the code to use 64 bit integers throughout.

An example `src/main.cpp` demonstrates usage of unweighted SGD based layout.

## todo

- Use STL templates rather than `uint64_t*` arrays.
