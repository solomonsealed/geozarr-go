package consistency

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
)

// The tests compare Go with Python libraries through oracle.py. With
// GEOZARR_PYTHON set they run it live; with -geozarr.record they also save
// its answers under testdata/recordings, one file per test, keyed by the
// request. Without Python they replay those answers, so every comparison
// also runs offline:
//
//	GEOZARR_PYTHON=.venv/bin/python go test ./consistency -geozarr.record

var (
	record   = flag.Bool("geozarr.record", false, "re-record testdata with the Python oracle")
	seedFlag = flag.Int64("geozarr.seed", 0, "seed of the random cases (live runs only; 0 keeps the recorded seed)")
	casesArg = flag.Int("geozarr.cases", 0, "number of random cases (live runs only; 0 keeps each test's default)")
)

const recordingsDir = "testdata/recordings"

// live is a running oracle.py.
type live struct {
	mu  sync.Mutex
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Reader
}

var (
	liveOnce sync.Once
	theLive  *live
	liveErr  error
)

func pythonPath() string {
	py := os.Getenv("GEOZARR_PYTHON")
	if py == "" || filepath.IsAbs(py) {
		return py
	}
	if abs, err := filepath.Abs(filepath.Join("..", py)); err == nil {
		if _, err := os.Stat(abs); err == nil {
			return abs
		}
	}
	return py
}

func startLive() (*live, error) {
	liveOnce.Do(func() {
		cmd := exec.Command(pythonPath(), "-I", "oracle.py")
		cmd.Stderr = os.Stderr
		in, err := cmd.StdinPipe()
		if err != nil {
			liveErr = err
			return
		}
		out, err := cmd.StdoutPipe()
		if err != nil {
			liveErr = err
			return
		}
		if err := cmd.Start(); err != nil {
			liveErr = err
			return
		}
		theLive = &live{cmd: cmd, in: in, out: bufio.NewReaderSize(out, 1<<20)}
	})
	return theLive, liveErr
}

func (l *live) call(line []byte) ([]byte, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, err := l.in.Write(append(line, '\n')); err != nil {
		return nil, err
	}
	resp, err := l.out.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("oracle: %w", err)
	}
	return resp, nil
}

// pyError is an exception raised in the oracle.
type pyError struct {
	Message string `json:"error"`
	Type    string `json:"type"`
}

func (e *pyError) Error() string { return "python: " + e.Message }

// oracle answers one test's requests, live or from its recording.
type oracle struct {
	t      testing.TB
	name   string
	live   *live
	mu     sync.Mutex
	answer map[string]json.RawMessage // recorded answers (replay), or new ones (record)
	calls  int
}

// isLive reports whether the comparison runs against Python itself.
func (o *oracle) isLive() bool { return o.live != nil }

// pythonOracle returns the oracle for t's top-level test: live if
// GEOZARR_PYTHON is set, otherwise replaying the test's recording (and
// skipping the test if there is none).
func pythonOracle(t testing.TB) *oracle {
	t.Helper()
	name, _, _ := strings.Cut(t.Name(), "/")
	o := &oracle{t: t, name: name, answer: map[string]json.RawMessage{}}
	if os.Getenv("GEOZARR_PYTHON") != "" {
		l, err := startLive()
		if err != nil {
			t.Fatalf("starting the oracle: %v", err)
		}
		o.live = l
		if *record {
			t.Cleanup(o.save)
		}
		return o
	}
	if *record {
		t.Fatal("-geozarr.record needs GEOZARR_PYTHON")
	}
	f, err := os.Open(o.path())
	if err != nil {
		t.Skipf("no recording of %s (set GEOZARR_PYTHON to compare with Python live): %v", name, err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	// Answers may hold Python's NaN and Infinity tokens; decoding them
	// reads the strings pythonTokens makes of them, as decodeResponse does
	// with live answers.
	if err := json.Unmarshal(pythonTokens(b), &o.answer); err != nil {
		t.Fatalf("reading %s: %v", o.path(), err)
	}
	return o
}

func (o *oracle) path() string {
	return filepath.Join(recordingsDir, o.name+".json.gz")
}

// save writes the answers of a recording run.
func (o *oracle) save() {
	if o.t.Failed() {
		o.t.Logf("not saving %s: the test failed", o.path())
		return
	}
	if len(o.answer) == 0 {
		return // skipped, or asked nothing
	}
	keys := make([]string, 0, len(o.answer))
	for k := range o.answer {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var buf bytes.Buffer
	buf.WriteString("{")
	for i, k := range keys {
		if i > 0 {
			buf.WriteString(",\n")
		}
		kb, _ := json.Marshal(k)
		buf.Write(kb)
		buf.WriteString(":")
		buf.Write(o.answer[k])
	}
	buf.WriteString("}\n")
	if err := os.MkdirAll(recordingsDir, 0o755); err != nil {
		o.t.Error(err)
		return
	}
	var gz bytes.Buffer
	zw, _ := gzip.NewWriterLevel(&gz, gzip.BestCompression)
	zw.Write(buf.Bytes())
	zw.Close()
	if err := os.WriteFile(o.path(), gz.Bytes(), 0o644); err != nil {
		o.t.Error(err)
	}
}

// call runs one operation; result receives the "ok" value. A Python
// exception comes back as a *pyError.
func (o *oracle) call(op string, req map[string]any, result any) error {
	if req == nil {
		req = map[string]any{}
	}
	req["op"] = op
	line, err := marshalRequest(req)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(line)
	key := hex.EncodeToString(sum[:16])
	var resp []byte
	if o.live != nil {
		if resp, err = o.live.call(line); err != nil {
			return err
		}
		resp = bytes.TrimSpace(resp)
		if *record {
			o.mu.Lock()
			o.answer[key] = append(json.RawMessage(nil), resp...)
			o.mu.Unlock()
		}
	} else {
		r, ok := o.answer[key]
		if !ok {
			return fmt.Errorf("no recorded answer to %s %.200s (re-record with GEOZARR_PYTHON and -geozarr.record)", op, line)
		}
		resp = r
	}
	o.mu.Lock()
	o.calls++
	o.mu.Unlock()
	var r struct {
		OK json.RawMessage `json:"ok"`
		pyError
	}
	if err := decodeResponse(resp, &r); err != nil {
		return fmt.Errorf("oracle: bad response %.200q: %w", resp, err)
	}
	if r.Message != "" {
		return &r.pyError
	}
	if result != nil {
		return decodeResponse(r.OK, result)
	}
	return nil
}

// marshalRequest encodes a request deterministically (sorted keys), so
// that the same request always has the same recording key.
func marshalRequest(req map[string]any) ([]byte, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	// Re-encode through a decoded value: encoding/json sorts map keys,
	// and this also canonicalizes json.RawMessage values.
	var v any
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	return json.Marshal(v)
}

// decodeResponse decodes JSON written by Python, which may contain bare
// NaN and Infinity tokens.
func decodeResponse(b []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	if err := dec.Decode(v); err == nil {
		return nil
	}
	fixed := pythonTokens(b)
	dec = json.NewDecoder(bytes.NewReader(fixed))
	dec.UseNumber()
	return dec.Decode(v)
}

// pythonTokens replaces Python's NaN and Infinity tokens outside strings
// with strings ("NaN", "Infinity", "-Infinity"); fields that can hold them
// decode those strings.
func pythonTokens(b []byte) []byte {
	var out bytes.Buffer
	inString := false
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inString {
			out.WriteByte(c)
			if c == '\\' && i+1 < len(b) {
				i++
				out.WriteByte(b[i])
			} else if c == '"' {
				inString = false
			}
			continue
		}
		switch {
		case c == '"':
			inString = true
			out.WriteByte(c)
		case bytes.HasPrefix(b[i:], []byte("NaN")):
			out.WriteString(`"NaN"`)
			i += 2
		case bytes.HasPrefix(b[i:], []byte("-Infinity")):
			out.WriteString(`"-Infinity"`)
			i += 8
		case bytes.HasPrefix(b[i:], []byte("Infinity")):
			out.WriteString(`"Infinity"`)
			i += 7
		default:
			out.WriteByte(c)
		}
	}
	return out.Bytes()
}

// summary counts comparisons for the line TestMain prints.
var summary struct {
	sync.Mutex
	counts map[string]int
}

func count(what string, n int) {
	summary.Lock()
	defer summary.Unlock()
	if summary.counts == nil {
		summary.counts = map[string]int{}
	}
	summary.counts[what] += n
}

func TestMain(m *testing.M) {
	code := m.Run()
	if len(summary.counts) > 0 {
		keys := make([]string, 0, len(summary.counts))
		for k := range summary.counts {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var parts []string
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%d %s", summary.counts[k], k))
		}
		mode := "replayed recordings"
		if theLive != nil {
			mode = "live"
		}
		fmt.Printf("consistency (%s): %s\n", mode, strings.Join(parts, ", "))
	}
	if theLive != nil {
		theLive.call([]byte(`{"op":"exit"}`))
		theLive.in.Close()
		theLive.cmd.Wait()
	}
	os.Exit(code)
}

// TestOracleVersions records the versions the comparison ran against.
func TestOracleVersions(t *testing.T) {
	o := pythonOracle(t)
	var v map[string]any
	if err := o.call("versions", nil, &v); err != nil {
		t.Fatal(err)
	}
	t.Logf("comparing against %v", v)
}

// seed returns the seed of a test's random cases: def unless a live run
// sets -geozarr.seed.
func seed(o *oracle, def int64) int64 {
	if o.isLive() && *seedFlag != 0 {
		return *seedFlag
	}
	return def
}

// cases returns the number of random cases: def unless a live run sets
// -geozarr.cases.
func cases(o *oracle, def int) int {
	if o.isLive() && *casesArg > 0 {
		return *casesArg
	}
	return def
}
