package com.fingerprint.recognization;

import android.content.Context;
import android.os.Handler;
import android.support.v4.hardware.fingerprint.FingerprintManagerCompat;
import android.support.v4.os.CancellationSignal;
import android.text.TextUtils;

import com.fingerprint.recognization.util.AESUtil;
import com.fingerprint.recognization.util.CryptoObjectHelper;
import com.fingerprint.recognization.util.FingerAuthCallback;
import com.fingerprint.recognization.util.FingerResultCode;
import com.fingerprint.recognization.util.SharedPreferenceUtil;

public enum FingerprintRecognize {
    INSTENCE;
    private Context context;
    private FingerprintManagerCompat mFingerprintManager;
    private CancellationSignal signal;

    public Context getContext() {
        return context;
    }

    public int init(Context context) {
        this.context = context;
        mFingerprintManager = FingerprintManagerCompat.from(context);
        if (!mFingerprintManager.isHardwareDetected()) {
            return FingerResultCode.CODE_INIT_FAIL_HARDWARE;
        } else if (!mFingerprintManager.hasEnrolledFingerprints()) {
            return FingerResultCode.CODE_INIT_FAIL_ENROLLED;
        }
        return FingerResultCode.CODE_INIT_SUCESS;
    }

    public boolean isTurnOn() {
        String loginInfo = AESUtil.decrypt(SharedPreferenceUtil.getInstance(FingerprintRecognize.INSTENCE.getContext())
                .getString(SharedPreferenceUtil.LOGIN_INFO_KEY));
        String loginUrl = AESUtil.decrypt(SharedPreferenceUtil.getInstance(FingerprintRecognize.INSTENCE.getContext())
                .getString(SharedPreferenceUtil.LOGIN_URL_KEY));
        if (TextUtils.isEmpty(loginInfo) || TextUtils.isEmpty(loginUrl)) {
            return false;
        }
        return true;
    }

    public int turnOn(String loginUrl, String loginInfo) {
        if (context == null) {
            return FingerResultCode.CODE_TURNON_FAIL_LACK_CONTEXT;
        }
        if (TextUtils.isEmpty(loginInfo) || TextUtils.isEmpty(loginUrl)) {
            return FingerResultCode.CODE_TURNON_FAIL_LACK_PARAMS;
        }
        SharedPreferenceUtil.getInstance(context).saveString(SharedPreferenceUtil.LOGIN_INFO_KEY,
                AESUtil.encrypt(loginInfo));
        SharedPreferenceUtil.getInstance(context).saveString(SharedPreferenceUtil.LOGIN_URL_KEY,
                AESUtil.encrypt(loginUrl));
        return FingerResultCode.CODE_TURNON_SUCESS;
    }

    public void detect(Handler handler) {
        if (handler == null) {
            return;
        }
        if (!isTurnOn()) {
            handler.obtainMessage(FingerResultCode.CODE_ERROR_NEVER_TRUNON).sendToTarget();
            return;
        }
        try {
            CryptoObjectHelper cryptoObjectHelper = new CryptoObjectHelper();
            if (signal == null) {
                signal = new CancellationSignal();
            }
            mFingerprintManager.authenticate(cryptoObjectHelper.buildCryptoObject(), 0,
                    signal, new FingerAuthCallback(handler), null);
        } catch (Exception e) {
            handler.obtainMessage(FingerResultCode.CODE_ERROR_UNKNOWN).sendToTarget();
        }

    }

    public void turnOff() {
        SharedPreferenceUtil.getInstance(context).clearKeys();
    }


}
