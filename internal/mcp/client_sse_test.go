package mcp

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// startFakeSSE returns a Client whose readSSE is already running, consuming
// the given data lines as an SSE body.
func startFakeSSE(t *testing.T, dataLines []string) *Client {
	t.Helper()
	c := NewClient("http://unused")
	go c.readSSE(&nopCloser{Reader: strings.NewReader(sseBody(dataLines))})
	return c
}

func sseBody(dataLines []string) string {
	var b strings.Builder
	for _, l := range dataLines {
		b.WriteString("data: " + l + "\n\n")
	}
	return b.String()
}

type nopCloser struct{ *strings.Reader }

func (n *nopCloser) Close() error { return nil }

// TestReadSSEDropsAfterGraceWindowWhenFull verifies the bounded blocking send:
// when msgCh is full and nobody drains it, readSSE must give up after the
// grace window (not block forever and stall the HTTP body), dropping only
// after waiting. This pins the behavior change from immediate-drop (select
// default) to bounded-block introduced for R3-BUG35.
func TestReadSSEDropsAfterGraceWindowWhenFull(t *testing.T) {
	if testing.Short() {
		t.Skip("timing-sensitive")
	}
	c := NewClient("http://unused")
	// Fill msgCh to capacity; no receiver exists.
	for i := 0; i < cap(c.msgCh); i++ {
		c.msgCh <- []byte(`{"fill":true}`)
	}

	start := time.Now()
	done := make(chan struct{})
	go func() {
		defer close(done)
		c.readSSE(&nopCloser{Reader: strings.NewReader(sseBody([]string{
			`{"jsonrpc":"2.0","id":1}`,
			`{"jsonrpc":"2.0","id":2}`,
		}))})
	}()

	select {
	case <-done:
		elapsed := time.Since(start)
		if elapsed < sendGraceWindow {
			t.Fatalf("readSSE returned before grace window elapsed (%v < %v): message was dropped immediately", elapsed, sendGraceWindow)
		}
		if elapsed > 10*sendGraceWindow {
			t.Fatalf("readSSE took far too long: %v", elapsed)
		}
	case <-time.After(10 * sendGraceWindow):
		t.Fatal("readSSE still blocked after 10x grace window: unbounded blocking send regression")
	}
}

// TestReadSSEDeliversWhenDrained verifies that messages are delivered
// end-to-end when a consumer is actively reading from msgCh — the common path
// must not be degraded by the drop logic or the reused timer.
func TestReadSSEDeliversWhenDrained(t *testing.T) {
	const n = 200 // well past channel capacity to exercise Stop+Reset reuse
	lines := make([]string, n)
	for i := range lines {
		lines[i] = fmt.Sprintf(`{"seq":%d}`, i)
	}
	c := startFakeSSE(t, lines)

	for i := 0; i < n; i++ {
		select {
		case got := <-c.msgCh:
			want := fmt.Sprintf(`{"seq":%d}`, i)
			if string(got) != want {
				t.Fatalf("message %d out of order or corrupted: got %s want %s", i, got, want)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for message %d of %d", i, n)
		}
	}
}
