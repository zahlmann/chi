CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -D_POSIX_C_SOURCE=200809L
PKG_CONFIG ?= pkg-config

CURL_CFLAGS ?= $(shell $(PKG_CONFIG) --cflags libcurl 2>/dev/null)
CURL_LIBS ?= $(shell $(PKG_CONFIG) --libs libcurl 2>/dev/null)

.PHONY: all clean check-libcurl

all: chi

check-libcurl:
	@if [ -n "$(strip $(CURL_LIBS))" ]; then \
		exit 0; \
	fi
	@if ! command -v "$(PKG_CONFIG)" >/dev/null 2>&1; then \
		echo "$(PKG_CONFIG) not found." >&2; \
		echo "Install pkg-config on Debian/Ubuntu or pkgconf-pkg-config on Fedora/RHEL." >&2; \
		echo "Or pass CURL_CFLAGS=... CURL_LIBS=... to make." >&2; \
		exit 1; \
	fi
	@if [ -z "$(strip $(CURL_LIBS))" ]; then \
		echo "libcurl development files not found via $(PKG_CONFIG)." >&2; \
		echo "Install libcurl4-openssl-dev on Debian/Ubuntu or libcurl-devel on Fedora/RHEL." >&2; \
		exit 1; \
	fi

chi: chi.c apply_patch.c apply_patch.h | check-libcurl
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CURL_CFLAGS) chi.c apply_patch.c $(LDFLAGS) $(CURL_LIBS) -o $@

clean:
	rm -f chi
