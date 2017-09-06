package com.fingerprint.recognization.util;

public class FingerResultCode {
    public static final int CODE_INIT_SUCESS = 1100;
    public static final int CODE_INIT_FAIL_HARDWARE = 1201;
    public static final int CODE_INIT_FAIL_ENROLLED = 1202;

    public static final int CODE_TURNON_SUCESS = 2100;
    public static final int CODE_TURNON_FAIL_LACK_CONTEXT = 2201;
    public static final int CODE_TURNON_FAIL_LACK_PARAMS = 2202;


    public static final int CODE_AUTH_SUCCESS = 3000;
    public static final int CODE_AUTH_FAILED = 32001;
    public static final int CODE_AUTH_ERROR = 3202;
    public static final int CODE_AUTH_HELP = 3203;
    public static final int CODE_ERROR_UNKNOWN = 3301;
    public static final int CODE_ERROR_NEVER_TRUNON = 3302;
}
