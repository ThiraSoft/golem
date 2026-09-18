package tha3

// NumParams is the length of a pose.
const NumParams = 45

// Indices into a pose, in the order the demo's pose_parameters.py builds it.
// Parameters that come in pairs are left then right.
const (
	EyebrowTroubledLeft = iota
	EyebrowTroubledRight
	EyebrowAngryLeft
	EyebrowAngryRight
	EyebrowLoweredLeft
	EyebrowLoweredRight
	EyebrowRaisedLeft
	EyebrowRaisedRight
	EyebrowHappyLeft
	EyebrowHappyRight
	EyebrowSeriousLeft
	EyebrowSeriousRight
	EyeWinkLeft
	EyeWinkRight
	EyeHappyWinkLeft
	EyeHappyWinkRight
	EyeSurprisedLeft
	EyeSurprisedRight
	EyeRelaxedLeft
	EyeRelaxedRight
	EyeUnimpressedLeft
	EyeUnimpressedRight
	EyeRaisedLowerEyelidLeft
	EyeRaisedLowerEyelidRight
	IrisSmallLeft
	IrisSmallRight
	MouthAaa
	MouthIii
	MouthUuu
	MouthEee
	MouthOoo
	MouthDelta
	MouthLoweredCornerLeft
	MouthLoweredCornerRight
	MouthRaisedCornerLeft
	MouthRaisedCornerRight
	MouthSmirk
	IrisRotationX
	IrisRotationY
	HeadX
	HeadY
	NeckZ
	BodyY
	BodyZ
	Breathing
)

// Where the three groups of networks cut the pose.
const (
	eyebrowParams  = 12
	faceParamsEnd  = eyebrowParams + 27
	rotationParams = NumParams - faceParamsEnd
)
