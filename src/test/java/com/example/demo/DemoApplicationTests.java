package com.example.demo;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class DemoApplicationTests {

    @Test
    void contextLoads() {
        int a=1;
        char b='1';
        String c="1";
        if (a==b){
            System.out.println("相等");
        }else {
            System.out.println("不相等");
        }
        if (c.equals(a)){
            System.out.println("相等");
        }else {
            System.out.println("不相等");
        }

    }

}
