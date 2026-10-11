"""Host-side resource-managed function worker (scientific runtime may be a container)."""

from bindocracy.functions.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
