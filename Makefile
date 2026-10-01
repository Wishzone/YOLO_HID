.PHONY: all clean rebuild package check-deps test

all:
	$(MAKE) -C src all

check-deps:
	$(MAKE) -C src check-deps

test:
	$(MAKE) -C src test

clean:
	$(MAKE) -C src clean

rebuild:
	$(MAKE) clean
	$(MAKE) all

package: all
	bash scripts/package.sh
