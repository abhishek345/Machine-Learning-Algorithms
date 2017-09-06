package com.fingerprint.recognization.util;

import android.annotation.SuppressLint;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

/**
 * AES加密解密工具包
 *
 * @author chenqsh
 */
@SuppressLint("NewApi")
public class AESUtil {
    private static final String AES = "AES";//AES 加密
    private static final String CIPHER_TRANSFORMATION = "AES/ECB/PKCS5Padding";
    private static final String CHARSET = "utf-8";
    public static final String AES_KEY = "7908ODE5BEDF89D8";

    public static String encrypt(String text) {

        if (text == null || text.equals("")) {
            return "";
        }
        String result = "";
        try {
            Cipher cipher = Cipher.getInstance(CIPHER_TRANSFORMATION);
            byte[] keyBytes = new byte[16];
            byte[] b = AES_KEY.getBytes(CHARSET);
            int len = b.length;
            if (len > keyBytes.length)
                len = keyBytes.length;
            System.arraycopy(b, 0, keyBytes, 0, len);
            SecretKeySpec keySpec = new SecretKeySpec(keyBytes, AES);
            cipher.init(Cipher.ENCRYPT_MODE, keySpec);
            byte[] results = cipher.doFinal(text.getBytes(CHARSET));
            result = Base64Util.encode(results);
        } catch (Exception e) {
        }
        return result;
    }

    public static String decrypt(String text) {
        if (text == null || text.equals("")) {
            return "";
        }
        String result = "";
        try {
            Cipher cipher = Cipher.getInstance(CIPHER_TRANSFORMATION);
            byte[] keyBytes = new byte[16];
            byte[] b = AES_KEY.getBytes(CHARSET);
            int len = b.length;
            if (len > keyBytes.length)
                len = keyBytes.length;
            System.arraycopy(b, 0, keyBytes, 0, len);
            SecretKeySpec keySpec = new SecretKeySpec(keyBytes, AES);
            cipher.init(Cipher.DECRYPT_MODE, keySpec);

            byte[] results = cipher.doFinal(Base64Util.decode(text));
            result = new String(results, CHARSET);
        } catch (Exception e) {
        }
        return result;
    }
}