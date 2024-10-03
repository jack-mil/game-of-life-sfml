#/usr/bin/env sh
clang-tidy src/*.[ch]pp -config='' -- -std=c++20 -I./build/_deps/sfml-src/include -I./third-party/ $@
