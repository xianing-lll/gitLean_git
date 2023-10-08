package com.example.demo;

import java.math.BigDecimal;

public class TestBook {
    public static void main(String[] args) {
        Double result;
        Double a= Double.valueOf(2000101),b= Double.valueOf(10),c= Double.valueOf(65),d= Double.valueOf(9999999);
        BigDecimal decimal = new BigDecimal(a/b/c);
        System.out.println(decimal);
        result= decimal.setScale(2,BigDecimal.ROUND_HALF_UP).doubleValue();
        System.out.println(result);
        System.out.println(d-result);
    }
}
