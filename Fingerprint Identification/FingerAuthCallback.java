package com.fingerprint.recognization.util;

import android.os.Handler;
import android.support.v4.hardware.fingerprint.FingerprintManagerCompat;

import com.fingerprint.recognization.FingerprintRecognize;

public class FingerAuthCallback extends FingerprintManagerCompat.AuthenticationCallback {

    private Handler handler = null;

    public FingerAuthCallback(Handler handler) {
        super();
        this.handler = handler;
    }

    @Override
    public void onAuthenticationError(int errMsgId, CharSequence errString) {
        super.onAuthenticationError(errMsgId, errString);

        if (handler != null) {
            handler.obtainMessage(FingerResultCode.CODE_AUTH_ERROR, errMsgId, 0).sendToTarget();
        }
    }

    @Override
    public void onAuthenticationHelp(int helpMsgId, CharSequence helpString) {
        super.onAuthenticationHelp(helpMsgId, helpString);
        if (handler != null) {
            handler.obtainMessage(FingerResultCode.CODE_AUTH_HELP, helpMsgId, 0).sendToTarget();
        }
    }

    @Override
    public void onAuthenticationSucceeded(FingerprintManagerCompat.AuthenticationResult result) {
        super.onAuthenticationSucceeded(result);
        String loginInfo = AESUtil.decrypt(SharedPreferenceUtil.getInstance(FingerprintRecognize.INSTENCE.getContext())
                .getString(SharedPreferenceUtil.LOGIN_INFO_KEY));
        String loginUrl = AESUtil.decrypt(SharedPreferenceUtil.getInstance(FingerprintRecognize.INSTENCE.getContext())
                .getString(SharedPreferenceUtil.LOGIN_URL_KEY));
        try {
            HttpUtil.doPostAsyn(loginUrl, loginInfo, new HttpUtil.HttpCallBack() {
                @Override
                public void onRequestComplete(String result) {
                    if (handler != null) {
                        handler.obtainMessage(FingerResultCode.CODE_AUTH_SUCCESS, 0, 0, result).sendToTarget();
                    }
                }
            });
        } catch (Exception e) {
            if (handler != null) {
                handler.obtainMessage(FingerResultCode.CODE_AUTH_SUCCESS).sendToTarget();
            }
        }
    }

    @Override
    public void onAuthenticationFailed() {
        super.onAuthenticationFailed();
        if (handler != null) {
            handler.obtainMessage(FingerResultCode.CODE_AUTH_FAILED).sendToTarget();
        }
    }
}
