# NVMeVirt — Module Management
#
# Targets:
#   make build          Build the NVMeVirt kernel module
#   make clean          Clean the NVMeVirt build tree
#   sudo make load      Load nvmev.ko (insmod with module parameters)
#   sudo make unload    Unload nvmev.ko
#   sudo make reload    Unload then load
#   make status         Show module and device status
#
# Configuration can be overridden:
#   sudo make load MEMMAP_START=0x1000000000 MEMMAP_SIZE=0x1000000000 CPUS=16,17-31

SHELL := /bin/bash

ROOT_DIR   := $(shell pwd)
NVMEV_DIR  := $(ROOT_DIR)/nvmevirt
NVMEV_KO   := $(NVMEV_DIR)/nvmev.ko

# NVMeVirt virtual PCI device address (domain 0x0001, bus 0x10)
NVMEV_BDF  := 0001:10:00.0

# BaM (bundles libnvm) — optional GPU-initiated NVMe driver that binds the virtual
# device INSTEAD of the stock nvme driver. Vendored as a submodule at control/vendor/bam.
# The kernel module (libnvm.ko) MUST be built with the kernel's compiler (gcc-9 here);
# the system default gcc is 13, so build-libnvm forces CC=$(LIBNVM_CC) for the module.
LIBNVM_DIR ?= $(ROOT_DIR)/vendor/bam
LIBNVM_KO  ?= $(LIBNVM_DIR)/build/module/libnvm.ko
LIBNVM_CC  ?= /usr/bin/gcc-9

# Default module parameters (64G reserved at physical address 64G)
MEMMAP_START ?= 68719476736
MEMMAP_SIZE  ?= 68719476736
CPUS         ?= 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31

# Timeout waiting for virtual PCI device to appear (seconds)
PCI_WAIT_TIMEOUT ?= 5

.PHONY: help build clean load unload reload status _check_root _check_ko \
        build-libnvm load-libnvm unload-libnvm reload-libnvm

help:
	@echo "Usage:"
	@echo "  make build          — Build the NVMeVirt kernel module"
	@echo "  make clean          — Clean the build tree"
	@echo "  sudo make load      — Load nvmev.ko"
	@echo "  sudo make unload    — Unload nvmev.ko"
	@echo "  sudo make reload    — Unload then load"
	@echo "  sudo make status    — Show module/device status"
	@echo ""
	@echo "  make build-libnvm   — Build libnvm/BaM (from $(LIBNVM_DIR))"
	@echo "  sudo make load-libnvm   — Unbind nvme from the virtual dev, load libnvm"
	@echo "  sudo make unload-libnvm — Unload libnvm, rebind nvme"
	@echo "  sudo make reload-libnvm — Unload then load libnvm"
	@echo ""
	@echo "Configuration (override via environment or command line):"
	@echo "  MEMMAP_START  Physical address of reserved memory  [$(MEMMAP_START)]"
	@echo "  MEMMAP_SIZE   Size of reserved memory              [$(MEMMAP_SIZE)]"
	@echo "  CPUS          CPU list: dispatcher,worker,worker,… [$(CPUS)]"

build:
	@echo "[BUILD] NVMeVirt kernel module"
	$(MAKE) -C $(NVMEV_DIR)
	@ls -lh $(NVMEV_KO)

clean:
	$(MAKE) -C $(NVMEV_DIR) clean

# ── libnvm / BaM on the virtual device ──────────────────────────────────────
# libnvm binds the NVMeVirt virtual device in place of the stock nvme driver,
# giving GPU-initiated NVMe access (BaM). Needs the OPEN nvidia driver (GPUDirect).
build-libnvm:
	@test -d $(LIBNVM_DIR) || { echo "Error: $(LIBNVM_DIR) not found — vendor it first:"; \
		echo "  git submodule add https://github.com/ZaidQureshi/bam control/vendor/bam"; \
		echo "  git submodule update --init --recursive control/vendor/bam"; exit 1; }
	@echo "[BUILD] BaM/libnvm in $(LIBNVM_DIR) (kernel module CC=$(LIBNVM_CC))"
	@mkdir -p $(LIBNVM_DIR)/build
	cd $(LIBNVM_DIR)/build && cmake ..
	@# libnvm.ko must match the kernel compiler (gcc-9); default gcc is 13 here.
	$(MAKE) -C $(LIBNVM_DIR)/build/module CC=$(LIBNVM_CC)
	@ls -lh $(LIBNVM_KO)
	@echo "[note] userspace lib (for BaM apps): make -C $(LIBNVM_DIR)/build libnvm"
	$(MAKE) -C $(LIBNVM_DIR)/build/ libnvm -j CC=$(LIBNVM_CC)
	$(MAKE) -C $(LIBNVM_DIR)/build/ benchmarks -j CC=$(LIBNVM_CC)


load-libnvm: _check_root
	@if ! lsmod | grep -q '^nvmev '; then \
		echo "Error: NVMeVirt not loaded — run 'sudo make load' first"; exit 1; fi
	@if [ ! -f $(LIBNVM_KO) ]; then \
		echo "Error: $(LIBNVM_KO) not found — run 'make build-libnvm'"; exit 1; fi
	@# Unbind the stock nvme driver from the virtual device so libnvm can claim it
	@if [ -L /sys/bus/pci/devices/$(NVMEV_BDF)/driver ]; then \
		DRV=$$(basename $$(readlink /sys/bus/pci/devices/$(NVMEV_BDF)/driver)); \
		echo "[UNBIND] $$DRV from $(NVMEV_BDF)"; \
		echo $(NVMEV_BDF) > /sys/bus/pci/devices/$(NVMEV_BDF)/driver/unbind 2>/dev/null || true; \
		sleep 0.5; \
	else echo "[SKIP] no driver bound to $(NVMEV_BDF)"; fi
	@echo "[LOAD] insmod $(LIBNVM_KO)"
	insmod $(LIBNVM_KO)
	@sleep 0.5
	@if ls /dev/libnvm* >/dev/null 2>&1; then \
		echo "[OK]   $$(ls /dev/libnvm*) ready"; chmod 666 /dev/libnvm* 2>/dev/null || true; \
	else echo "[WARN] /dev/libnvm* not found — check dmesg"; fi

unload-libnvm: _check_root
	@if lsmod | grep -q '^libnvm '; then echo "[UNLOAD] rmmod libnvm"; rmmod libnvm; \
	else echo "[SKIP]   libnvm not loaded"; fi
	@# Rebind the stock nvme driver to the virtual device
	@echo "[REBIND] nvme -> $(NVMEV_BDF)"
	@echo $(NVMEV_BDF) > /sys/bus/pci/drivers/nvme/bind 2>/dev/null || \
	 echo "         (rebind failed — 'echo $(NVMEV_BDF) > /sys/bus/pci/drivers_probe' or reload nvmev)"

reload-libnvm: _check_root
	@$(MAKE) --no-print-directory unload-libnvm
	@sleep 1
	@$(MAKE) --no-print-directory load-libnvm

clean-libnvm:
	@$(MAKE) -C $(LIBNVM_DIR)/build/module clean
	@$(MAKE) -C $(LIBNVM_DIR)/build/ clean
	rm -rf $(LIBNVM_DIR)/build

load: _check_root _check_ko
	@# ── Step 1: Load NVMeVirt ──────────────────────────────────────
	@if lsmod | grep -q '^nvmev '; then \
		echo "[SKIP] nvmev already loaded"; \
	else \
		echo "[LOAD] insmod nvmev.ko (memmap_start=$(MEMMAP_START), memmap_size=$(MEMMAP_SIZE))"; \
		insmod $(NVMEV_KO) \
			memmap_start=$(MEMMAP_START) \
			memmap_size=$(MEMMAP_SIZE) \
			cpus=$(CPUS); \
		echo "[OK]   nvmev loaded"; \
	fi
	@# ── Step 2: Wait for virtual PCI device ────────────────────────
	@echo -n "[WAIT] Waiting for PCI device $(NVMEV_BDF)"
	@for i in $$(seq 1 $(PCI_WAIT_TIMEOUT)); do \
		if [ -d /sys/bus/pci/devices/$(NVMEV_BDF) ]; then \
			echo " found"; \
			break; \
		fi; \
		echo -n "."; \
		sleep 1; \
		if [ $$i -eq $(PCI_WAIT_TIMEOUT) ]; then \
			echo " TIMEOUT"; \
			echo "[ERR]  PCI device $(NVMEV_BDF) did not appear within $(PCI_WAIT_TIMEOUT)s"; \
			exit 1; \
		fi; \
	done
	@echo ""
	@echo "=== Load complete ==="

unload: _check_root
	@if lsmod | grep -q '^nvmev '; then \
		echo "[UNLOAD] rmmod nvmev"; \
		rmmod nvmev; \
	else \
		echo "[SKIP]   nvmev not loaded"; \
	fi
	@echo ""
	@echo "=== Unload complete ==="

reload: _check_root
	@echo "=== Reload: unload → load ==="
	@echo ""
	@$(MAKE) --no-print-directory unload
	@echo ""
	@sleep 1
	@$(MAKE) --no-print-directory load

status:
	@echo "=== Module Status ==="
	@echo ""
	@echo "Loaded modules:"
	@lsmod | grep -E '^nvmev ' | awk '{printf "  %-16s %s (used by: %s)\n", $$1, $$2, $$4}' || echo "  (none)"
	@echo ""
	@echo "NVMeVirt PCI device ($(NVMEV_BDF)):"
	@if [ -d /sys/bus/pci/devices/$(NVMEV_BDF) ]; then \
		DRIVER="(none)"; \
		if [ -L /sys/bus/pci/devices/$(NVMEV_BDF)/driver ]; then \
			DRIVER=$$(basename $$(readlink /sys/bus/pci/devices/$(NVMEV_BDF)/driver)); \
		fi; \
		echo "  Present, driver=$$DRIVER"; \
	else \
		echo "  Not present"; \
	fi
	@echo ""
	@echo "NVMeVirt procfs:"
	@if [ -d /proc/nvmev ]; then \
		for f in /proc/nvmev/*; do \
			echo "  $$f: $$(head -1 $$f 2>/dev/null)"; \
		done; \
	else \
		echo "  (not available)"; \
	fi

_check_root:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: must run as root (use sudo make ...)"; \
		exit 1; \
	fi

_check_ko:
	@if [ ! -f $(NVMEV_KO) ]; then \
		echo "Error: $(NVMEV_KO) not found — run 'make build' first"; \
		exit 1; \
	fi
