# Xtensa LX7 PIE SIMD Reference

## Registers
- `q0-q7`: 128-bit SIMD vector registers
- `ACCX`, `ACCY`: accumulator registers
- `SAR`: shift amount register

## Load / Store (16-byte aligned)
```
EE.VLD.128.IP    qX, aY, imm    # Load 128 bits, post-increment aY by imm
EE.VST.128.IP    qX, aY, imm    # Store 128 bits, post-increment
EE.VLD.128.XP    qX, aY, aZ     # Load 128 bits, post-increment by register
```

## Bitwise Operations
```
EE.ANDQ    qZ, qX, qY    # 128-bit AND
EE.ORQ     qZ, qX, qY    # 128-bit OR
EE.XORQ    qZ, qX, qY    # 128-bit XOR
EE.NOTQ    qZ, qX         # 128-bit NOT
```

## INT8 Multiply-Accumulate
```
EE.VMULAS.S8.ACCX              qX, qY         # 16x INT8 signed MAC into ACCX
EE.VMULAS.S8.ACCX.LD.IP        qZ, aW, 16, qX, qY   # MAC + preload next vector
EE.VMULAS.S8.ACCX.LD.IP.QUP    qZ, aW, 16, qX, qY, qR, qS  # MAC + load + shift
```

## Data Movement
```
EE.MOVI.32.A    qX, aY, sel    # Move 32-bit lane from q-reg to scalar
EE.MOVI.32.Q    qX, aY, sel    # Move scalar into q-reg lane
EE.ZERO.Q       qX              # Zero a q-register
```

## Critical Notes

1. **ALL vector loads/stores require 16-byte alignment**. No unaligned access.
   Use `__attribute__((aligned(16)))` on all buffers.

2. **No hardware popcount**. Must use software (shift-and-mask or LUT).

3. **Ternary strategy**: Expand 2-bit ternary weights to INT8 {-1, 0, +1},
   then use `EE.VMULAS.S8.ACCX` — avoids popcount entirely.

4. **Register budget**: q0-q7 = 8 registers.
   Inner loop needs: weights, activations, expanded weights, scratch = 4 min.

5. **Pipeline**: Xtensa can dual-issue some instruction pairs.
   Overlap loads with computation using `LD.IP` variants.
