package gemma

// What the card actually holds, with the experts on it and off it.
//
// It is the claim the README has to be able to make. "A mixture larger than the
// card" is not shown by a model that fits: the 26B A4B is 12.8 GiB against
// sixteen, and it fits either way. What can be shown is how much of the card it
// stops needing — and that is the number that says how much larger a mixture
// could be, rather than asserting that one would run.
//
// **The measurement has to name the cache, and this test did not.** Since the
// cache landed, a plan left to itself spends whatever device memory is free on
// a copy of the experts a token keeps asking for, so GOLEM_MOE_EXPERTS_HOST=1
// alone fills the card back up — a hundred and twenty-eight slots a block here,
// which is the whole pool again — and the two measurements came out the same to
// within the noise of the desktop. That is the planner doing what the README
// says it does, not the feature failing. The floor is asked for by name:
// GOLEM_MOE_CACHE_SLOTS=0, the pool in host memory and none of it on the card.
//
// The figures come from the driver, through VK_EXT_memory_budget, which is why
// this could not have been written before the instance stopped asking for
// Vulkan 1.0 under the name of 1.1.

import (
	"os"
	"testing"
)

func TestExpertResidency(t *testing.T) {
	path := model26BPath(t)

	measure := func(env map[string]string) (uint64, error) {
		for _, k := range []string{"GOLEM_MOE_EXPERTS_HOST", "GOLEM_MOE_CACHE_SLOTS"} {
			if v, ok := env[k]; ok {
				t.Setenv(k, v)
			} else {
				os.Unsetenv(k)
			}
		}
		m, err := Open(path, 512)
		if err != nil {
			return 0, err
		}
		defer m.Close()
		d, err := m.device()
		if err != nil {
			return 0, err
		}
		before := d.DeviceLocalFree()
		if before == 0 {
			t.Skip("the driver will not report a memory budget")
		}
		if err := m.UseVulkanStack(); err != nil {
			return 0, err
		}
		return before - d.DeviceLocalFree(), nil
	}

	resident, err := measure(nil)
	if err != nil {
		t.Skipf("resident: %v", err)
	}
	// The floor: the pools in host memory and no cache in front of them. Not
	// GOLEM_MOE_EXPERTS_HOST alone, which leaves the planner free to spend the
	// card on cache — see the head of this file.
	inHost, err := measure(map[string]string{"GOLEM_MOE_EXPERTS_HOST": "1", "GOLEM_MOE_CACHE_SLOTS": "0"})
	if err != nil {
		t.Fatalf("experts in host memory: %v", err)
	}
	t.Logf("the card holds %d MiB with the experts on it, %d MiB with them in host memory and no cache",
		resident>>20, inHost>>20)
	if inHost >= resident {
		t.Fatalf("moving the experts off the card freed nothing: %d MiB against %d", inHost>>20, resident>>20)
	}
	t.Logf("so the experts are %d MiB of it, and what is left is the shared branches, the attention and the head",
		(resident-inHost)>>20)
	// The README's table says 13.6 GiB against 1.3, which is a tenth. Half is
	// the assertion, because what is free on the card moves with the desktop
	// and the point is the order of magnitude, not the digit.
	if inHost > resident/2 {
		t.Errorf("the experts are most of the card, so moving them off it should free most of it: %d MiB against %d", inHost>>20, resident>>20)
	}
}
