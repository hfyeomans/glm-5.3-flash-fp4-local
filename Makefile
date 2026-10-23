.PHONY: check plan doctor prepare start stop smoke
check:
	./scripts/check.sh
plan:
	./start.sh --dry-run
doctor:
	./doctor.sh
prepare:
	./prepare.sh
start:
	./start.sh
stop:
	./stop.sh
smoke:
	./smoke.sh --suite
