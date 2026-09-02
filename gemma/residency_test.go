package gemma

// What the card actually holds, with the experts on it and off it.
//
// It is the claim the README has to be able to make. "A mixture larger than the
// card" is not shown by a model that fits: the 26B A4B is 12.8 GiB against
// sixteen, and it fits either way. What can be shown is how much of the card it
// stops needing — and that is the number that says how much larger a mixture
// could be, rather than asserting that one would run.
//
// The figures come from the driver, through VK_EXT_memory_budget, which is why
// this could not have been written before the instance stopped asking for
// Vulkan 1.0 under the name of 1.1.

import (
	"os"
	"testing"
)

func TestExpertResidency(t *testing.T) {
	if testing.Short() {
		t.Skip("uploads the 26B twice")
	}
	path := model26BPath(t)

	measure := func(inHost bool) (uint64, error) {
		if inHost {
			t.Setenv("GOLEM_MOE_EXPERTS_HOST", "1")
		} else {
			os.Unsetenv("GOLEM_MOE_EXPERTS_HOST")
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

	resident, err := measure(false)
	if err != nil {
		t.Skipf("resident: %v", err)
	}
	inHost, err := measure(true)
	if err != nil {
		t.Fatalf("experts in host memory: %v", err)
	}
	t.Logf("the card holds %d MiB with the experts on it, %d MiB with them in host memory",
		resident>>20, inHost>>20)
	t.Logf("so the experts are %d MiB of it, and what is left is the shared branches, the attention and the head",
		(resident-inHost)>>20)
	if inHost >= resident {
		t.Fatalf("moving the experts off the card freed nothing: %d MiB against %d", inHost>>20, resident>>20)
	}
}
