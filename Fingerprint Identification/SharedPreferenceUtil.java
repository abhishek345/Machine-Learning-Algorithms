package com.fingerprint.recognization.util;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * Description: SharedPreference工具类(单例)。
 * 用法举例：SharedPreferenceUtil.INSTANCE.getString("xxx");
 *
 * @author qisheng.chen
 * @date 2015/10/12 11:38
 */
public class SharedPreferenceUtil {
    public static final String LOGIN_INFO_KEY = "login_INFO_KEY";
    public static final String LOGIN_URL_KEY = "login_URL_KEY";
    private static SharedPreferenceUtil instance;
    private final String PREF_NAME = "FingerRecognizeSP";
    protected SharedPreferences mSettings;
    protected SharedPreferences.Editor mEditor;
    private Context mContext;

    public static SharedPreferenceUtil getInstance(Context context) {
        if (instance == null) {
            instance = new SharedPreferenceUtil(context);
        }
        return instance;
    }

    // 初始化构造
    private SharedPreferenceUtil(Context context) {
        this.mContext = context;
        if (mContext == null) {
            return;
        }
        mSettings = mContext.getSharedPreferences(PREF_NAME, 0);
        mEditor = mSettings.edit();
    }


    public String getString(String key) {
        return mSettings.getString(key, "");
    }

    public String getString(String key, String defaultValue) {
        return mSettings.getString(key, defaultValue);
    }

    public void saveString(String key, String value) {
        mEditor.putString(key, value);
        mEditor.commit();
    }

    public boolean getBoolean(String key, boolean defValue) {
        return mSettings.getBoolean(key, defValue);
    }

    public void saveBoolean(String key, boolean value) {
        mEditor.putBoolean(key, value);
        mEditor.commit();
    }

    public int getInt(String key, int defValue) {
        return mSettings.getInt(key, defValue);
    }

    public void saveInt(String key, int value) {
        mEditor.putInt(key, value);
        mEditor.commit();
    }

    public long getLong(String key, long defValue) {
        return mSettings.getLong(key, defValue);
    }

    public void saveLong(String key, long value) {
        mEditor.putLong(key, value);
        mEditor.commit();
    }

    public void removeKey(String key) {
        mEditor.remove(key);
        mEditor.commit();
    }

    public void clearKeys() {
        mEditor.clear();
        mEditor.commit();
    }

    public boolean isContains(String key) {
        return mSettings.contains(key);
    }
}
