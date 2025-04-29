    .text
    .align 4

    // ------------------------------------------
    // void compute_dense_relu(
    //     float *in,      // x0
    //     float *weights, // x1
    //     float *bias,    // x2
    //     float *out,     // x3
    //     int    in_size, // x4
    //     int    out_size // x5
    // );
    .global _compute_dense_relu
_compute_dense_relu:
    mov     w6, #0              // i = 0
    fmov    s1, #0.0            // zero for ReLU

.L_dr_out:
    cmp     w6, w5
    bge     .L_dr_done

    eor     v0.16b, v0.16b, v0.16b   // acc = 0
    lsl     x9, x6, #2               // x9 = i * 4
    ldr     s4, [x2, x9]             // bias[i]

    // row_ptr = weights + i*in_size*4
    mul     x8, x6, x4
    lsl     x8, x8, #2
    add     x8, x1, x8

    mov     w7, #0              // j = 0

.L_dr_in:
    cmp     w7, w4
    bge     .L_dr_after

    lsl     x9, x7, #2           // x9 = j * 4
    ldr     q1, [x0, x9]         // load in[j..j+3]
    ldr     q2, [x8, x9]         // load w[j..j+3]
    fmla    v0.4s, v1.4s, v2.4s  // acc += in·w

    add     w7, w7, #4
    b       .L_dr_in

.L_dr_after:
    addv    s0, v0.4s           // s0 = sum(acc[0..3])
    fadd    s0, s0, s4          // + bias
    fmax    s0, s0, s1          // ReLU
    lsl     x9, x6, #2
    str     s0, [x3, x9]        // out[i] = s0

    add     w6, w6, #1
    b       .L_dr_out

.L_dr_done:
    ret


    // ------------------------------------------
    // void compute_dense(
    //     float *in,      // x0
    //     float *weights, // x1
    //     float *bias,    // x2
    //     float *out,     // x3
    //     int    in_size, // x4
    //     int    out_size // x5
    // );
    .global _compute_dense
_compute_dense:
    mov     w6, #0

.L_d_out:
    cmp     w6, w5
    bge     .L_d_done

    eor     v0.16b, v0.16b, v0.16b
    lsl     x9, x6, #2
    ldr     s4, [x2, x9]

    mul     x8, x6, x4
    lsl     x8, x8, #2
    add     x8, x1, x8

    mov     w7, #0

.L_d_in:
    cmp     w7, w4
    bge     .L_d_after

    lsl     x9, x7, #2
    ldr     q1, [x0, x9]
    ldr     q2, [x8, x9]
    fmla    v0.4s, v1.4s, v2.4s

    add     w7, w7, #4
    b       .L_d_in

.L_d_after:
    addv    s0, v0.4s
    fadd    s0, s0, s4
    lsl     x9, x6, #2
    str     s0, [x3, x9]

    add     w6, w6, #1
    b       .L_d_out

.L_d_done:
    ret


    // ------------------------------------------
    // float compute_mse(
    //     float *orig,  // x0
    //     float *recon, // x1
    //     int    n      // x2
    // );
    .global _compute_mse
_compute_mse:
    mov     w3, #0
    mov     x9, x0              // ptr_o = orig
    mov     x10, x1             // ptr_r = recon
    movi    v16.4s, #0          // acc_vec = 0

.L_mse:
    cmp     w3, w2
    bge     .L_mse_end

    ldr     q0, [x9]            // load orig[ j..j+3 ]
    ldr     q1, [x10]           // load recon[j..j+3]
    fsub    v2.4s, v0.4s, v1.4s // diff
    fmul    v3.4s, v2.4s, v2.4s // diff²
    fadd    v16.4s, v16.4s, v3.4s

    add     x9, x9, #16         // advance by 4 floats
    add     x10, x10, #16
    add     w3, w3, #4
    b       .L_mse

.L_mse_end:
    addv    s0, v16.4s          // sum(diff²)
    scvtf   s1, w2              // n → float
    fdiv    s0, s0, s1          // mean
    ret