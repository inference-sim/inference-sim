package workload

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

// failingWriter accepts nothing: every Write returns errWrite. closeErr is what Close returns.
type failingWriter struct {
	failWrite bool
	closeErr  error
	buf       bytes.Buffer
	closed    bool
}

var errWrite = errors.New("disk full")

func (w *failingWriter) Write(p []byte) (int, error) {
	if w.failWrite {
		return 0, errWrite
	}
	return w.buf.Write(p)
}

func (w *failingWriter) Close() error { w.closed = true; return w.closeErr }

func writerTestRecords() (*TraceHeader, []TraceRecord) {
	h := &TraceHeader{Version: 2, TimeUnit: "microseconds", Mode: "generated"}
	return h, []TraceRecord{{RequestID: 0, InputTokens: 10, OutputTokens: 5, ArrivalTimeUs: 1}}
}

// SIM-2, #661: the csv.Writer buffers every row, so a failing file is seen only at the
// flush. A trace whose rows never reached the file is not reported as exported.
func TestTraceV2Export_AFailingWriteIsReturned(t *testing.T) {
	h, recs := writerTestRecords()
	w := &failingWriter{failWrite: true}
	err := writeTraceV2Data(w, h, recs)
	if !errors.Is(err, errWrite) {
		t.Fatalf("a trace whose rows were never written was reported as exported: err = %v", err)
	}
	if !w.closed {
		t.Error("the trace data file was not closed after a failed write")
	}
}

// SIM-2, #661: a close that fails (on a network file system, the write-back) is the
// export's error.
func TestTraceV2Export_AFailingCloseIsReturned(t *testing.T) {
	h, recs := writerTestRecords()
	errClose := errors.New("close: input/output error")
	w := &failingWriter{closeErr: errClose}
	err := writeTraceV2Data(w, h, recs)
	if !errors.Is(err, errClose) {
		t.Fatalf("a failed close was not returned: err = %v", err)
	}
}

// The writer returns no error when every write, the flush and the close succeed, and writes
// the header row and each record.
func TestTraceV2Export_ASucceedingWriterGetsEveryRow(t *testing.T) {
	h, recs := writerTestRecords()
	w := &failingWriter{}
	if err := writeTraceV2Data(w, h, recs); err != nil {
		t.Fatalf("writeTraceV2Data: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(w.buf.String()), "\n")
	if len(lines) != 1+len(recs) || !strings.HasPrefix(lines[0], "request_id,") {
		t.Errorf("wrote %d lines, want a header and %d rows:\n%s", len(lines), len(recs), w.buf.String())
	}
}
