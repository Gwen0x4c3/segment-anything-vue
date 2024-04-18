function throttle(func, delay) {
    let timer = null; // 定时器变量

    return function() {
        const context = this; // 保存this指向
        const args = arguments; // 保存参数列表

        if (!timer) {
            timer = setTimeout(() => {
                func.apply(context, args); // 调用原始函数并传入上下文和参数
                clearTimeout(timer); // 清除计时器
                timer = null; // 重置计时器为null
            }, delay);
        }
    };
}
export default throttle