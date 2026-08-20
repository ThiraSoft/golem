//go:build amd64

#include "textflag.h"

// func spinPause()
//
// Eight PAUSE instructions, the same count as the arithmetic loop this
// replaces. On this machine PAUSE is some forty cycles, so the turn is longer
// than the loop it replaces — which is the point: a waiting core should look at
// the counter rarely and cheaply.
TEXT ·spinPause(SB), NOSPLIT|NOFRAME, $0-0
	PAUSE
	PAUSE
	PAUSE
	PAUSE
	PAUSE
	PAUSE
	PAUSE
	PAUSE
	RET
