from pymilvus import MilvusClient
def test2():
    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    # ① 在末尾追加元素
    # 7
    # ② 在下标
    # 2
    # 位置插入
    # 100
    # ③ 删除第一个
    # 1
    # ④ 列表升序排序
    # ⑤ 统计元素
    # 1
    # 出现次数
    nums.append(7)
    nums.insert(2, 100)
    nums.remove(1)
    nums.sort()
    count = nums.count(1)

    fruits = ["苹果", "香蕉", "橙子"]
    # 用两种方式遍历：
    # 方式
    # 1：直接
    # for 循环元素
    #     方式
    #     2：带下标遍历（用
    #     enumerate）
    for f in fruits:
        print(f"  {f}")
    for i, f in enumerate(fruits):
        print(f"  [{i}] {f}")

    t = (2, 4, 6, 8, 10)
    # ① 获取第
    # 3
    # 个元素
    # ② 判断
    # 5
    # 是否在元组中
    # ③ 统计
    # 4
    # 出现次数
    # ④ 把元组转为列表
    a = t[2]
    b = t.index(5)
    c = t.count(4)
    lst = list(t)

    user = {"name": "李四", "age": 22}
    # ① 新增键值对
    # gender: "男"
    # ② 修改年龄为
    # 25
    # ③ 用
    # setdefault
    # 加
    # city: "北京"，已存在则不修改
    user["gender"] = "男"
    user["age"] = 25
    result = user.setdefault("city", "北京")

    # 沿用上面 user 字典
    # ① 用 [] 取 name
    # ② 用 get 取 age，取不到给默认值 18
    # ③ 用 get 取不存在的键 score，默认返回 0
    print(f"  {user['name']}{user['age']}")
    user.get("age", 18)
    user.get("score", 0)

    stu = {"id": 101, "name": "王五", "score": 88, "class": "一班"}
    # ① 删除
    #
    # class 键
    #     ② 随机删除最后一组键值对（popitem）
    #     ③ 清空整个字典（clear）
    del stu["class"]
    key, value = stu.popitem()
    stu.clear()

    book = {"书名": "Python入门", "价格": 59, "作者": "张三"}
    # 分别遍历：所有 key、所有 value、所有键值对
    for k in book.keys():
        print(k)
        print(book[k])
    for k, v in book.items():
        print(k, v)

    li = [2, 2, 3, 3, 4, 4, 5]
    # 写代码去重，生成新列表。
    list(set(li))

    keys = ["name", "age", "city"]
    vals = ["赵六", 28, "广州"]
    # 把两个列表合并成一个字典。
    dict(zip(keys, vals))

def demo():
    # 必须和 config.py 中 get_milvus_config() 返回的 uri 一模一样
    uri = "http://114.132.151.31:19530"   # 替换成你的实际地址
    client = MilvusClient(uri=uri)

    # 列出所有集合
    collections = client.list_collections()
    print("当前集合列表:", collections)

    # 检查目标集合是否存在
    exists = client.has_collection("customer_service_kb")
    print(f"customer_service_kb 存在: {exists}")

    # 如果存在，删除它
    if exists:
        client.drop_collection("customer_service_kb")
        print("已删除")
    else:
        print("集合已经不存在")


if __name__ == "__main__":
    demo()