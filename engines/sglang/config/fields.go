// Original to blis-sglang.

package config

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// fieldRef is one ServerArgs field: its SGLang name, the site of its declaration, and a pointer
// to it in a given ServerArgs.
type fieldRef struct {
	name string
	site string
	ref  func(*ServerArgs) any
}

var fieldIndex = func() map[string]fieldRef {
	m := make(map[string]fieldRef, len(fieldTable))
	for _, f := range fieldTable {
		m[f.name] = f
	}
	return m
}()

// FieldNames lists every ServerArgs field by its SGLang name, sorted.
func FieldNames() []string {
	out := make([]string, 0, len(fieldTable))
	for _, f := range fieldTable {
		out = append(out, f.name)
	}
	sort.Strings(out)
	return out
}

// IsField reports whether name is a ServerArgs field of the pinned release.
func IsField(name string) bool {
	_, ok := fieldIndex[name]
	return ok
}

func ptr[T any](v T) *T { return &v }

// set stores v in the field called name. v is either a Go value of the field's type (or of its
// base type, for a pointer field) or a value decoded from JSON: a float64 with no fraction for an
// int, a []any for a list. nil is Python's None.
func (s *ServerArgs) set(name string, v any) error {
	f, ok := fieldIndex[name]
	if !ok {
		return fmt.Errorf("%s is not a field of SGLang's ServerArgs", name)
	}
	bad := func(want string) error {
		return fmt.Errorf("%s: %v (%T) is not %s", name, v, v, want)
	}
	switch p := f.ref(s).(type) {
	case *bool:
		b, ok := v.(bool)
		if !ok {
			return bad("a bool")
		}
		*p = b
	case *int:
		n, ok := asInt(v)
		if !ok {
			return bad("an int")
		}
		*p = n
	case *float64:
		x, ok := asFloat(v)
		if !ok {
			return bad("a number")
		}
		*p = x
	case *string:
		str, ok := v.(string)
		if !ok {
			return bad("a string")
		}
		*p = str
	case **bool:
		if v == nil {
			*p = nil
			return nil
		}
		if q, ok := v.(*bool); ok {
			*p = clonePtr(q)
			return nil
		}
		b, ok := v.(bool)
		if !ok {
			return bad("a bool or None")
		}
		*p = &b
	case **int:
		if v == nil {
			*p = nil
			return nil
		}
		if q, ok := v.(*int); ok {
			*p = clonePtr(q)
			return nil
		}
		n, ok := asInt(v)
		if !ok {
			return bad("an int or None")
		}
		*p = &n
	case **float64:
		if v == nil {
			*p = nil
			return nil
		}
		if q, ok := v.(*float64); ok {
			*p = clonePtr(q)
			return nil
		}
		x, ok := asFloat(v)
		if !ok {
			return bad("a number or None")
		}
		*p = &x
	case **string:
		if v == nil {
			*p = nil
			return nil
		}
		if q, ok := v.(*string); ok {
			*p = clonePtr(q)
			return nil
		}
		str, ok := v.(string)
		if !ok {
			return bad("a string or None")
		}
		*p = &str
	case *[]string:
		l, ok := asList(v, func(x any) (string, bool) { s, ok := x.(string); return s, ok })
		if !ok {
			return bad("a list of strings or None")
		}
		*p = l
	case *[]int:
		l, ok := asList(v, asInt)
		if !ok {
			return bad("a list of ints or None")
		}
		*p = l
	case *[]float64:
		l, ok := asList(v, asFloat)
		if !ok {
			return bad("a list of numbers or None")
		}
		*p = l
	case **CudaGraphConfig:
		if v == nil {
			*p = nil
			return nil
		}
		c, ok := v.(*CudaGraphConfig)
		if !ok {
			return bad("a CudaGraphConfig or None")
		}
		*p = c.clone()
	case *any:
		*p = v
	default:
		panic(fmt.Sprintf("config: field %s has an unhandled Go type %T", name, p))
	}
	return nil
}

// value is the field's value as JSON holds it, which is how SGLang's resolution records it
// (oracle/run_serverargs.py): None is nil, a number is a float64, a list is a []any.
func (s *ServerArgs) value(name string) any {
	f, ok := fieldIndex[name]
	if !ok {
		panic("config: no field " + name)
	}
	b, err := json.Marshal(f.ref(s))
	if err != nil {
		panic(fmt.Sprintf("config: %s: %v", name, err))
	}
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		panic(fmt.Sprintf("config: %s: %v", name, err))
	}
	return v
}

// differs reports whether a field holds different values in a and b.
func differs(a, b *ServerArgs, name string) bool {
	x, _ := json.Marshal(a.value(name))
	y, _ := json.Marshal(b.value(name))
	return string(x) != string(y)
}

func clonePtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		if n == math.Trunc(n) && math.Abs(n) < 1<<53 {
			return int(n), true
		}
	case json.Number:
		i, err := n.Int64()
		return int(i), err == nil
	}
	return 0, false
}

func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		x, err := n.Float64()
		return x, err == nil
	}
	return 0, false
}

func asList[T any](v any, elem func(any) (T, bool)) ([]T, bool) {
	switch l := v.(type) {
	case nil:
		return nil, true
	case []T:
		return append([]T{}, l...), true
	case []any:
		out := make([]T, 0, len(l))
		for _, x := range l {
			e, ok := elem(x)
			if !ok {
				return nil, false
			}
			out = append(out, e)
		}
		return out, true
	}
	return nil, false
}
