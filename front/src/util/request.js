import axios from "axios";
import { ElMessage } from "element-plus";

// 请求拦截  设置统一header
axios.interceptors.request.use(
    config => {
        // if (localStorage.eleToken)
        //     config.headers.Authorization = localStorage.eleToken;
        return config;
    },
    error => {
        return Promise.reject(error);
    }
);

// 响应拦截  401 token过期处理
axios.interceptors.response.use(
    response => {
        console.log('response: ', response)
        if (response.data.success != null && !response.data.success) {
            return Promise.reject(response.data)
        }
        return response.data;
    },
    error => {
        console.log('error: ', error)
        ElMessage.error(' ');

        // const { status } = error.response;
        // if (status == 401) {
        //     ElMessage.error("token值无效，请重新登录");
        //     // 清除token
        //     localStorage.removeItem("eleToken");
        //     // 页面跳转
        //     router.push("/login");
        // }

        return Promise.reject(error);
    }
);

export default axios;
