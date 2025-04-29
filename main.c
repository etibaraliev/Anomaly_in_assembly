#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

// tell the compiler these routines live in our .s file:
extern void   compute_dense_relu(float *in,   float *weights, float *bias,
                                 float *out,  int in_size,   int out_size);
extern void   compute_dense     (float *in,   float *weights, float *bias,
                                 float *out,  int in_size,   int out_size);
extern float  compute_mse       (float *orig, float *recon,   int n);

// helper to malloc-and-check
static void *must_alloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "ERROR: malloc(%zu) failed\n", bytes);
        exit(1);
    }
    return p;
}

int main(void) {
    // Example dimensions:
    const int in_size     = 1024;
    const int hidden_size =  256;
    const float threshold = 0.05f;

    // Allocate your buffers
    float *input  = must_alloc(sizeof *input * in_size);
    float *enc_w  = must_alloc(sizeof *enc_w * hidden_size * in_size);
    float *enc_b  = must_alloc(sizeof *enc_b * hidden_size);
    float *hidden = must_alloc(sizeof *hidden * hidden_size);
    float *dec_w  = must_alloc(sizeof *dec_w * in_size * hidden_size);
    float *dec_b  = must_alloc(sizeof *dec_b * in_size);
    float *recon  = must_alloc(sizeof *recon * in_size);

    // ------------------------------------------------------------------------
    // 1) Initialize with a toy random autoencoder (for testing)
    //    Replace these loops with real model-loading for your trained weights!
    srand((unsigned)time(NULL));
    // random input in [-1, +1]
    for (int i = 0; i < in_size; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    // small random encoder weights, zero biases
    for (int i = 0; i < hidden_size * in_size; i++) {
        enc_w[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
    for (int i = 0; i < hidden_size; i++) {
        enc_b[i] = 0.0f;
    }
    // small random decoder weights, zero biases
    for (int i = 0; i < in_size * hidden_size; i++) {
        dec_w[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
    for (int i = 0; i < in_size; i++) {
        dec_b[i] = 0.0f;
    }
    // ------------------------------------------------------------------------

    // 2) Run the autoencoder
    compute_dense_relu(input,
                       enc_w, enc_b,
                       hidden,
                       /* in_size */ in_size,
                       /* out_size */ hidden_size);

    compute_dense(hidden,
                  dec_w, dec_b,
                  recon,
                  /* in_size */ hidden_size,
                  /* out_size */ in_size);

    // 3) Compute MSE & anomaly flag
    float mse = compute_mse(input, recon, in_size);
    bool anomaly = (mse > threshold);

    printf("MSE = %f → %s\n", mse, anomaly ? "ANOMALY!" : "normal");

    // clean up
    free(input);
    free(enc_w);
    free(enc_b);
    free(hidden);
    free(dec_w);
    free(dec_b);
    free(recon);

    return anomaly ? 1 : 0;
}