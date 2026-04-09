#include "openssl-aes.h"

extern "C" {

int aesni_set_encrypt_key(const unsigned char *userKey, int bits, AES_KEY *key)
{
    return AES_set_encrypt_key(userKey, bits, key);
}

int aesni_set_decrypt_key(const unsigned char *userKey, int bits, AES_KEY *key)
{
    return AES_set_decrypt_key(userKey, bits, key);
}

void aesni_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY *key)
{
    AES_encrypt(in, out, key);
}

void aesni_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY *key)
{
    AES_decrypt(in, out, key);
}

void aesni_ecb_encrypt(const unsigned char *in, unsigned char *out, size_t length, const AES_KEY *key, int enc)
{
    AES_ecb_encrypt(in, out, key, enc);
}

void aesni_cbc_encrypt(const unsigned char *in, unsigned char *out, size_t length, const AES_KEY *key, unsigned char *ivec, int enc)
{
    AES_cbc_encrypt(in, out, length, key, ivec, enc);
}

} // extern "C"
