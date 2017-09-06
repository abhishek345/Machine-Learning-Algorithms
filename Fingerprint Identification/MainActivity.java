package com.fingerprint.sample;

import android.hardware.fingerprint.FingerprintManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.fingerprint.recognization.FingerprintRecognize;
import com.fingerprint.recognization.util.FingerResultCode;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private static final String TAG = "MainActivity";
    private Button turnOnBtn, turnOffBtn, turnStateBtn, detectBtn;
    private LinearLayout fingerprintLayout;
    private TextView fingerText;
    private TextView resultText;
    private final String loginUrl = "http://121.42.35.18:8998/pchat-manager/user/login";
    private final String loginParams = "{\"userNo\":\"005025\",\"password\":\"ify+/BDZ9NxvnPQUqL0p3wsoDbEv1G1sGQV/jKujwAge6BPkeFecUv1wKWPS1vlhwN4yjo+l112cpqz6SOFpbw==\"}";

    Handler handler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            switch (msg.what) {
                //没有开启指纹登录
                case FingerResultCode.CODE_ERROR_NEVER_TRUNON:
                    fingerprintLayout.setVisibility(View.GONE);
                    resultText.setText("尚未开启指纹登录，请先开启");
                    break;
                //指纹登录成功
                case FingerResultCode.CODE_AUTH_SUCCESS:
                    fingerText.setText("指纹识别成功");
                    resultText.setText("登录结果：" + msg.obj);
                    break;
                //指纹登录失败 指纹识别结果显示不在本机指纹库中
                case FingerResultCode.CODE_AUTH_FAILED:
                    fingerText.setText("请使用正确的手指");
                    break;
                //指纹识别出现错误
                case FingerResultCode.CODE_AUTH_ERROR:
                    handleErrorCode(msg.arg1);
                    break;
                //指纹识别出现帮助信息
                case FingerResultCode.CODE_AUTH_HELP:
                    handleHelpCode(msg.arg1);
                    break;
                //未知错误
                case FingerResultCode.CODE_ERROR_UNKNOWN:
                    fingerprintLayout.setVisibility(View.GONE);
                    resultText.setText("未知错误");
                    break;

            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initView();
        initFingerRecognize();
    }


    /**
     * 初始化指纹检测
     */
    private void initFingerRecognize() {
        //在调用其他方法前必须调用过init方法 建议应用启动或者页面渲染前调用init
        int initResult = FingerprintRecognize.INSTENCE.init(MainActivity.this);
        switch (initResult) {
            //初始化成功
            case FingerResultCode.CODE_INIT_SUCESS:
                resultText.setText("指纹登录初始化成功");
                switchBtnEnable(true);
                break;
            //初始化失败 硬件不支持指纹识别
            case FingerResultCode.CODE_INIT_FAIL_HARDWARE:
                resultText.setText("初始化失败 您的手机硬件不支持指纹识别");
                switchBtnEnable(false);
                break;
            //初始化失败 系统内没有指纹数据 需要先设置指纹
            case FingerResultCode.CODE_INIT_FAIL_ENROLLED:
                resultText.setText("初始化失败 系统内没有指纹数据，请您先设置指纹");
                switchBtnEnable(false);
                break;
        }
    }

    /**
     * 开启指纹识别
     */
    private void turnOnFinger() {
        //开启指纹登录 传入登录链接和登录参数 指纹解锁后自动登录
        int resultTurnOn = FingerprintRecognize.INSTENCE.turnOn(loginUrl, loginParams);
        switch (resultTurnOn) {
            //开启指纹登录设置成功
            case FingerResultCode.CODE_TURNON_SUCESS:
                resultText.setText("指纹登录开启成功！");
                detectBtn.setEnabled(true);
                break;
            //指纹登录设置失败，没有先调用初始化
            case FingerResultCode.CODE_TURNON_FAIL_LACK_CONTEXT:
                resultText.setText("指纹登录设置失败，没有先初始化。");
                break;
            //指纹登录设置失败，登录参数传入有误
            case FingerResultCode.CODE_TURNON_FAIL_LACK_PARAMS:
                resultText.setText("指纹登录设置失败，登录参数有误。");
                break;

        }
    }

    private void handleHelpCode(int code) {
        switch (code) {
            //获得的指纹图像很好
            case FingerprintManager.FINGERPRINT_ACQUIRED_GOOD:
                fingerText.setText("获得的指纹图像很好");
                break;
            //得到的指纹图像质量差 一般是因为传感器上有污垢
            case FingerprintManager.FINGERPRINT_ACQUIRED_IMAGER_DIRTY:
                fingerText.setText("手指或者指纹识别器上有污垢，请擦干净后重试");
                break;
            //得到的指纹图像质量差 可能是因为传感器上有污垢或手指太干燥
            case FingerprintManager.FINGERPRINT_ACQUIRED_INSUFFICIENT:
                fingerText.setText("手指或者指纹识别器上有污垢，请擦干净后重试");
                break;
            //只检测到部分指纹图像 一般因为手指未压紧
            case FingerprintManager.FINGERPRINT_ACQUIRED_PARTIAL:
                fingerText.setText("请压紧手指");
                break;
            //手指移动过快
            case FingerprintManager.FINGERPRINT_ACQUIRED_TOO_FAST:
                fingerText.setText("手指移动过快，请慢点");
                break;
            //手指没有移动  对于某些指纹传感器需要移动识别
            case FingerprintManager.FINGERPRINT_ACQUIRED_TOO_SLOW:
                fingerText.setText("请动动手指");
                break;
        }
    }

    private void handleErrorCode(int code) {
        switch (code) {
            //传感器被系统禁用 一般为用户切换/设备锁定/其他系统操作造成的指纹检测挂起
            case FingerprintManager.FINGERPRINT_ERROR_CANCELED:
                fingerText.setText("请稍后重试");
                break;
            //硬件不可用
            case FingerprintManager.FINGERPRINT_ERROR_HW_UNAVAILABLE:
                fingerText.setText("请重试");
                break;
            //错误次数过多
            case FingerprintManager.FINGERPRINT_ERROR_LOCKOUT:
                fingerText.setText("错误次数过多，请30秒后重试");
                break;
            //存储空间不足
            case FingerprintManager.FINGERPRINT_ERROR_NO_SPACE:
                fingerText.setText("内存不足，无法使用指纹登录");
                break;
            //超时 30秒内没有完成指纹登录
            case FingerprintManager.FINGERPRINT_ERROR_TIMEOUT:
                fingerText.setText("超时，请重新开始");
                break;
            //传感器错误 无法处理该指纹图像
            case FingerprintManager.FINGERPRINT_ERROR_UNABLE_TO_PROCESS:
                fingerText.setText("很抱歉，请重试");
                break;
        }
    }

    private void initView() {
        turnOnBtn = (Button) findViewById(R.id.btn_turn_on);
        turnOffBtn = (Button) findViewById(R.id.btn_turn_off);
        turnStateBtn = (Button) findViewById(R.id.btn_state);
        detectBtn = (Button) findViewById(R.id.btn_detect);
        turnOnBtn.setOnClickListener(this);
        turnOffBtn.setOnClickListener(this);
        turnStateBtn.setOnClickListener(this);
        detectBtn.setOnClickListener(this);
        fingerprintLayout = (LinearLayout) findViewById(R.id.layout_fingerprint);
        fingerText = (TextView) findViewById(R.id.result_fingerprint);
        resultText = (TextView) findViewById(R.id.text_result);
    }

    @Override
    public void onClick(View v) {
        resultText.setText("");
        switch (v.getId()) {
            case R.id.btn_turn_on:
                turnOnFinger();
                break;
            case R.id.btn_turn_off:
                //关闭指纹登录
                FingerprintRecognize.INSTENCE.turnOff();
                detectBtn.setEnabled(false);
                break;
            case R.id.btn_state:
                //读取本地是否开启了指纹登录
                boolean isTurnOn = FingerprintRecognize.INSTENCE.isTurnOn();
                if (isTurnOn) {
                    resultText.setText("指纹登录是开启状态，可以使用指纹登录");
                    detectBtn.setEnabled(true);
                } else {
                    resultText.setText("未开启指纹登录，请先开启指纹登录");
                    detectBtn.setEnabled(false);
                }
                break;
            case R.id.btn_detect:
                //进入指纹识别状态 传入Handler对象接收/处理结果
                FingerprintRecognize.INSTENCE.detect(handler);
                fingerprintLayout.setVisibility(View.VISIBLE);
                resultText.setText("请把指纹放在指纹识别器上");
                break;
        }

    }

    private void switchBtnEnable(boolean isEnable) {
        turnOnBtn.setEnabled(isEnable);
        turnOffBtn.setEnabled(isEnable);
        turnStateBtn.setEnabled(isEnable);
        detectBtn.setEnabled(FingerprintRecognize.INSTENCE.isTurnOn());
    }
}
