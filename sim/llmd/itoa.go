package llmd

// digitPairs holds "00" to "99" back to back.
const digitPairs = "0001020304050607080910111213141516171819" +
	"2021222324252627282930313233343536373839" +
	"4041424344454647484950515253545556575859" +
	"6061626364656667686970717273747576777879" +
	"8081828384858687888990919293949596979899"

// appendDec appends the decimal digits of v, the bytes strconv.AppendUint writes. Token IDs are
// short, so it writes the digits straight into b's spare room instead of through a scratch array.
func appendDec(b []byte, v uint32) []byte {
	n := 1
	switch {
	case v >= 1000000000:
		n = 10
	case v >= 100000000:
		n = 9
	case v >= 10000000:
		n = 8
	case v >= 1000000:
		n = 7
	case v >= 100000:
		n = 6
	case v >= 10000:
		n = 5
	case v >= 1000:
		n = 4
	case v >= 100:
		n = 3
	case v >= 10:
		n = 2
	}
	l := len(b)
	if cap(b)-l < n {
		b = append(b, make([]byte, n)...)
	} else {
		b = b[:l+n] // every byte of the n is written below
	}
	i := l + n
	for v >= 100 {
		q := v / 100
		r := (v - q*100) * 2
		i -= 2
		b[i], b[i+1] = digitPairs[r], digitPairs[r+1]
		v = q
	}
	if v >= 10 {
		r := v * 2
		b[i-2], b[i-1] = digitPairs[r], digitPairs[r+1]
	} else {
		b[i-1] = byte('0' + v)
	}
	return b
}
