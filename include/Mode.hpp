/*
Author: Jackson Miller
Class: ECE6122 A
Last Date Modified: 2024-10-10

Description:
Just a simple enum used in all files. Put here to prevent dependency loops
*/
#pragma once
/** Represent the possible CUDA memory modes */
enum class Mode { Normal, Pinned, Managed };
