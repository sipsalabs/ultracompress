"""UltraCompress — public CLI.

This distribution provides pack-structure and download-integrity checking
plus project information. It deliberately does NOT contain the compression
or reconstruction methodology: that is patent-pending and is not shipped in
the public package. Full cryptographically verifiable reconstruction (a
deterministic decode to the SHA-256-pinned validated artifact) is
performed by Sipsa Labs under engagement.
"""

__version__ = "0.6.24"
__all__ = ["__version__"]
