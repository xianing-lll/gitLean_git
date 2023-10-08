package com.example.demo;

import javax.swing.*;
import java.lang.reflect.Array;
import java.util.*;

public class SnakeGame {
    public int longestPalindromeSubseq(String s) {
        int[][] dp=new int[s.length()][s.length()];
        for (int i = s.length()-1; i >=0 ; i--) {
            dp[i][i]=1;
            for (int j = i+1; j < s.length(); j++) {
                if (s.charAt(i)==s.charAt(j)){
                    dp[i][j]=dp[i+1][j-1]+2;
                }else {
                    dp[i][j]=Math.max(dp[i+1][j],dp[i][j-1]);
                }
            }
        }

        return dp[0][s.length()-1];
    }
    public int minInsertions(String s) {
        //原长度减去最长回文子序列的长度即为需要插入字符的长度
        int n = s.length();
        int[] f = new int[n];
        for (int i = n-1; i >= 0; i--) {
            f[i]=1;
            int pre=0;
            for (int j = i+1; j < n; j++) {
                int temp=f[j];
                if (s.charAt(i)==s.charAt(j)){
                    f[j]=pre+2;
                }else {
                    f[j]=Math.max(f[j],f[j-1]);
                }
                pre=temp;
            }
        }
        return n-f[n-1];
    }

    public int getMoneyAmount(int n) {
        int[][] f = new int[n + 2][n + 2];
        for (int i = n-1; i >0 ; i--) {
            for (int j = i+1; j <=n; j++) {
                int temp=Integer.MAX_VALUE;
                for (int k = i; k <=j; k++) {
                    temp=Math.min(temp,k+Math.max(f[i][k-1],f[k+1][j]));
                }
                f[i][j]=temp;
            }
        }
        return f[0][n];
    }
    public int rob(int[] nums) {
        int n=nums.length;
        int[] dp = new int[n];
        dp[0]=nums[0];
        if (n>1){
            dp[1]=Math.max(nums[0],nums[1]);
        }
        for (int i = 2; i < n; i++) {
            //f[k]=Max{f[k-1],H[k-1]+f[k-2]}
            dp[i]=Math.max(dp[i-1],nums[i]+dp[i-2]);
        }
        return dp[n-1];
    }

    public int findTargetSumWays(int[] nums, int target) {
        //p:正符号数之和 s:nums之和  s-p:负符号数之和
        //所以，p-(s-p)=target 2p=target+s p=(garget+s)/2
        int n=nums.length;
        int sum=0;
        for (int i = 0; i < n; i++) {
            sum+=nums[i];
        }
        target+=sum;
        if (target<0 || target%2==1){//边界判断 target必须大于0，同时必须为0数，否则的话返回0
            return 0;
        }
        //正符号数之和
        target=target/2;
        int[][] f = new int[n+1][target + 1];//二维代表当第i个数为正或者负时，当前target剩余的target值
        f[0][0]=1;//当选择第0个数且target等于0是只有一中选择那就是不选
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c <=target ; ++c) {
                if (nums[i]>c){//当第i个值大于剩余的target值C时
                    f[i+1][c]=f[i][c];
                }else {
                    f[i+1][c]=f[i][c]+f[i][c-nums[i]];
                }
            }
        }
        return f[n][target];

    }
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        //初始化dp
        Arrays.fill(dp,amount+1);
        dp[0]=0;//当amont等于0时，没有金币组可以组成他
        for (int i = 1; i < amount+1; i++) {
            for (int j = 0; j < coins.length; j++) {
                //F(i)=min j=0…n−1 F(i−cj​)+1
                if (i>=coins[j]){
                    dp[i]=Math.min(dp[i],dp[i-coins[j]]+1);
                }
            }
        }
        return dp[amount]==(amount+1)?-1:dp[amount];
    }
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        //边界情况考虑
        if (n<2){
            return false;
        }
        int sum=0, maxNum=0;
        for (int num:nums) {
            sum+=num;
            maxNum=Math.max(maxNum,num);
        }
        //sum是奇数也不行
        if (sum%2==1) return false;
        //如果sum是偶数则要判断sum/2=target 是否大于等于Max[i],如果小于，则return false;
        if (sum/2<maxNum) return false;
        int target=sum/2;
        //dp[i][j] 表示从数组的 [0,i] 下标范围内选取若干个正整数（可以是 0 个），是否存在一种选取方案使得被选取的正整数的和等于 j
        boolean[][] dp = new boolean[n][target+1];
        //1、如果不选择任何正整数，则被选取的正整数为0.因此对于任何0<=i<n,都有dp[i][0]=true;
        for (int i = 0; i < n; i++) {
            dp[i][0]=true;
        }
        //2、当i==0时，只有一个正整数nums[0]可以被选取，因此 dp[0][nums[0]]==true;
        dp[0][nums[0]]=true;
        for (int i = 1; i < n; i++) {
            for (int j = 1; j <=target; j++) {
                if (j>=nums[i]){
                    dp[i][j]=dp[i-1][j-nums[i]]|dp[i-1][j];
                }else {//j<num[i]
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return dp[n-1][target];
    }
    public int change(int amount, int[] coins) {
        int n = coins.length;
        //d[i][j]表示从前i中硬币中凑出金额为j的的硬币的组合数/方案数6
        int[][] dp = new int[n+1][amount+1];
        //当金额j=0，硬币数量为0时，dp[0][0]=1
        dp[0][0]=1;
        //当金额不为空，硬币为空时 dp[i][0]=0,不用初始化，默认为0
        for (int i = 1; i <= n; i++) { //遍历硬币
            for (int j = 0; j <= amount; j++) {//遍历金额/背包
                //容量有限，无法选择第i个硬币
                if (j<coins[i-1]){
                    dp[i][j]=dp[i-1][j];
                }else {//可以选择第i个硬币
                    dp[i][j]=dp[i-1][j]+dp[i][j-coins[i-1]];
                }

            }
        }
        for (int i = 1; i <= n; i++) { //遍历硬币
            for (int j = 0; j <= amount; j++) {//遍历金额/背包
                //容量有限，无法选择第i个硬币
                System.out.print(dp[i][j]+" ");

            }
            System.out.println();
        }
        return dp[n][amount];
    }

    public int change2(int amount, int[] coins) {
        int n = coins.length;
        //d[i][j]表示从前i中硬币中凑出金额为j的的硬币的组合数/方案数6
        int[][] dp = new int[n+1][amount+1];
        //当金额j=0，硬币数量为0时，dp[0][0]=1
        dp[0][0]=1;
        //当金额不为空，硬币为空时 dp[i][0]=0,不用初始化，默认为0
        for (int i = 1; i <= n; i++) { //遍历硬币
            for (int j = 0; j <= amount; j++) {//遍历金额/背包
                dp[i][j]=dp[i-1][j];
                for (int k = 1; k*coins[i-1] <= j; k++) {
                    dp[i][j]+=dp[i-1][j-k*coins[i-1]];
                }
            }
        }

        return dp[n][amount];
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        //dp[i]:字符串前i个字符组成的字符串能否空格拆分成若干个由字典组成的单词
        boolean[] dp=new boolean[s.length()+1];
        dp[0]=true;//前0个字符可以由前0个字符组成
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j]&&wordDict.contains(s.substring(j,i))){
                    dp[i]=true;
                }
            }
        }
        return dp[s.length()];
        
    }
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target+1];
        dp[0]=1;
        for (int i = 1; i <= target; i++) {
            for (int num:nums) {
                if (i>=num){
                    dp[i]+=dp[i-num];
                }
            }
        }
        return dp[target];
    }
    public int findMaxForm(String[] strs, int m, int n) {
        int len=strs.length;
        int[][][] dp = new int[len+1][m+1][n+1];
        for (int i = 1; i <= len ; i++) {
            int[] count = countZeroAndOne(strs[i - 1]);
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= n ; k++) {
                    //先把上一行抄下来，假设不选择当前考虑字符串
                    dp[i][j][k]=dp[i-1][j][k];
                    int zeros = count[0];
                    int ones = count[1];
                    if (j>=zeros && k>= ones){//当0和1都能装进容器j和k时
                        dp[i][j][k]=Math.max(dp[i-1][j][k],dp[i-1][j-zeros][k-ones]+1);
                    }
                }
            }
        }
        return dp[len][m][n];
    }
    private int[] countZeroAndOne(String str){
        int[] ints = new int[2];
        for (char c:str.toCharArray()) {
            ints[c-'0']++;
        }
        return ints;
    }
    public int climbStairs(int n) {
        //斐波那契数列
        int[] dp = new int[n + 3];
        dp[0]=0;
        dp[1]=1;
        dp[2]=1;
        for (int i = 3; i <= n; i++) {
            dp[i]=dp[i-1]+dp[i-2]+dp[i-3];
        }
        return dp[n];
    }
    public int minCostClimbingStairs(int[] cost) {
        int[] dp = new int[cost.length + 1];
        dp[0]=0;
        dp[1]=0;
        for (int i = 2; i <= cost.length ; i++) {
            dp[i]=Math.min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2]);
        }
        return dp[cost.length];
    }
    public int deleteAndEarn(int[] nums) {
        int max=0;
        for (int num:nums) {
            max = Math.max(max, num);
        }
        int[] daJiaJieShe = new int[max + 1];
        for (int num:nums) {
            daJiaJieShe[num]++;
        }
        int[] dp = new int[max+1];
        dp[1]=daJiaJieShe[1];
        for (int i = 2; i <=max ; i++) {
            dp[i]=Math.max(dp[i-1],dp[i-2]+i*daJiaJieShe[i]);
        }
        return dp[max];
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m=obstacleGrid.length;
        int n=obstacleGrid[0].length;
        int[][] dp = new int[m+1][n+1];
        if (obstacleGrid[0][0]==0){
            dp[0][0]=1;
        }else {
            return dp[m-1][n-1];
        }
        for (int i = 1; i < m; i++) {
            if (obstacleGrid[i][0]==0){
                dp[i][0]=dp[i-1][0];
            }else {
                break;
            }
        }
        for (int i = 1; i < n; i++) {
            if (obstacleGrid[0][i]==0){
                dp[0][i]=dp[0][i-1];
            }else {
                break;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if(obstacleGrid[i][j]==0){
                    dp[i][j]=dp[i-1][j]+dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];
    }
    public int minimumTotal(List<List<Integer>> triangle) {
        int m=triangle.size(),n=triangle.get(triangle.size()).size();
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int maxValue = Integer.MAX_VALUE;
                dp[i][j]=maxValue;
            }
        }
        dp[0][0]=triangle.get(0).get(0);
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < triangle.get(i).size(); j++) {
                dp[i][j]=Math.min(j>0?dp[i-1][j-1]:Integer.MAX_VALUE,dp[i-1][j])+triangle.get(i).get(j);
            }
        }
        int min=dp[m-1][0];
        for (int i = 1; i < n; i++) {
            min = Math.min(min, dp[m - 1][i]);
        }
        return min;
    }
    public int minFallingPathSum(int[][] matrix) {
        int m=matrix.length,n=matrix[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) {
            dp[0][i]=matrix[0][i];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j]=Math.min(Math.min(dp[i-1][j],j>0?dp[i-1][j-1]:Integer.MAX_VALUE),j<n-1?dp[i-1][j+1]:Integer.MAX_VALUE)+matrix[i][j];
            }
        }
        int min=dp[m-1][0];
        for (int i = 1; i < n; i++) {
            min=Math.min(min,dp[m-1][i]);
        }

        return min;
    }
    public int maximalSquare(char[][] matrix) {
        int m=matrix.length,n=matrix[0].length;
        int[][] dp = new int[m][n];
        int maxEdge=0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j]=='1'){
                    dp[i][j]=1;
                    maxEdge=1;
                }

            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i-1][j-1]=='1'&&matrix[i-1][j]=='1'&&matrix[i][j-1]=='1'&&matrix[i][j]=='1'){
                    dp[i][j]+=Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1]));
                    maxEdge=Math.max(maxEdge,dp[i][j]);
                }
            }
        }
        return maxEdge*maxEdge;

    }

    public int minDistance(String word1, String word2) {
        int m=word1.length(),n=word2.length();
        int[][] dp = new int[m + 1][n + 1];
        //初始化
        dp[0][0]=0;
        for (int i = 1; i <= m; i++) {
            dp[i][0]=i;
        }
        for (int i = 1; i <= n; i++) {
            dp[0][i]=i;
        }
        for (int i = 1; i <=m ; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i-1)==word2.charAt(j-1)){
                    dp[i][j]=dp[i-1][j-1];
                }else {
                    dp[i][j]=Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1]))+1;
                }
            }
        }
        return dp[m][n];


    }

    public int minimumDeleteSum(String s1, String s2) {

        int m=s1.length(),n=s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m ; i++) {
            for (int j = 1; j <= n ; j++) {
                if (s1.charAt(i-1)==s2.charAt(j-1)){
                    dp[i][j]=dp[i-1][j-1]+(int)s1.charAt(i-1);
                }else {
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        int sum=0;
        for (int i = 0; i < m; i++) {
            sum+=(int)s1.charAt(i);
        }
        for (int i = 0; i < n; i++) {
            sum+=(int)s2.charAt(i);
        }
        return sum-2*dp[m][n];

    }

    public int numDistinct(String s, String t) {
        int m=s.length(),n=t.length();
        int[][] dp = new int[m + 1][n + 1];
        int mod= (int) (Math.pow(10,9)+7);
        //初始化
        for (int i = 0; i <= m; i++) {
            dp[i][0]=1;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i-1)!=t.charAt(j-1)){//当s[i]==t[j],并且选择s[i]时
                    dp[i][j]=dp[i-1][j];//s[i]!=t[j],
                }else {
                    dp[i][j]=(dp[i-1][j]+dp[i-1][j-1])%mod;//另一种是s[i]==t[j],有两种情况，一种是选s[i],另一种是即便s[i]=t[j]也不选
                }
            }
        }
        return dp[m][n];
    }
    public int lengthOfLIS(int[] nums) {
        int m=nums.length;
        int[] dp = new int[m + 1];
        //初始化
        for (int i = 0; i < m; i++) {
            dp[i]=1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j]<nums[i]){
                    dp[i]= Math.max(dp[j] + 1, dp[i]);
                }
            }
        }
        int max=dp[0];
        for (int i = 1; i < m; i++) {
            max = Math.max(max, dp[i]);
        }
        return max;
    }
    public int findNumberOfLIS(int[] nums) {
        int m=nums.length;
        int[] dp = new int[m];
        int[] counts = new int[m];
        for (int i = 0; i < m; i++) {
            dp[i]=1;
            counts[i]=1;
            for (int j = 0; j < i; j++) {
                if (nums[j]<nums[i]){
                    if (dp[j] + 1>dp[i]){
                        dp[i]=dp[j]+1;
                        counts[i]=counts[j];
                    }else if (dp[j] + 1==dp[i]){
                        counts[i]+=counts[j];
                    }
                }
            }
        }
        int max=dp[0];
        for (int i = 1; i < m; i++) {
            if(dp[i]>max){
                max=dp[i];
            }
        }
        int count=0;
        for (int i = 0; i < m; i++) {
            if (dp[i]==max){
                count+=counts[i];
            }
        }
        return count;
    }
    public int findLongestChain(int[][] pairs) {
        Arrays.sort(pairs, (o1, o2) -> o1[0]-o2[0]);
        int m=pairs.length;
        int[] dp = new int[m + 1];
        Arrays.fill(dp,1);
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < i; j++) {
                if (pairs[j][1]<pairs[i][0]){
                    dp[i]=Math.max(dp[i],dp[j]+1);
                }
            }
        }
        int max=dp[0];
        for (int i = 1; i < m; i++) {
            max=Math.max(max,dp[i]);
        }
        return max;
    }
    public int longestSubsequence(int[] arr, int difference) {
        HashMap<Integer, Integer> dp = new HashMap<>();
        int max=0;
        for (int v:arr) {
            dp.put(v,dp.getOrDefault(v-difference,0)+1);
            max=Math.max(max,dp.get(v));
        }
        return max;
    }
    public int longestArithSeqLength(int[] nums) {
        int m=nums.length;
        int[][] dp = new int[m][1001];
        int max=0;
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < i; j++) {
                int diff = nums[i] - nums[j] + 500;
                dp[i][diff]=dp[j][diff]+1;
                max=Math.max(max,dp[i][diff]);
            }
        }
        return max+1;
    }

    public int maxEnvelopes(int[][] envelopes) {
        Arrays.sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0]!=o2[0]){
                    return o1[0]-o2[0];
                }else {
                    return o2[1]-o1[1];
                }
            }
        });
        int m=envelopes.length;
        ArrayList<Integer> dp = new ArrayList<>();
        dp.add(envelopes[0][1]);//初始值为排序后信奉的第一个高度
        for (int i = 1; i < m; i++) {
            int l=0,r=dp.size()-1;
            int index=-1;
            while (l<=r){
                int mid=l+(r-l)/2;
                if (dp.get(mid)>=envelopes[i][1]){//向左
                    // 我们要找的是dp数组中第一个大于等于当前h的位置,因此在这里更新index值
                    index=mid;
                    r=mid-1;
                }else {//向右
                    l=mid+1;
                }
            }
            if (index==-1){
                dp.add(envelopes[i][1]);
            }else {
                dp.set(index,envelopes[i][1]);
            }
        }
        return dp.size();

    }

    public int[] longestObstacleCourseAtEachPosition(int[] obstacles) {
        int n = obstacles.length;
        int[] dp = new int[n+1],ans = new int[n];
        dp[0]=obstacles[0];
        ans[0]=1;
        int j=0;
        for (int i = 1; i < n; i++) {
            if (obstacles[i]>=dp[j]){
                dp[++j]=obstacles[i];
                ans[i]=j+1;
            }else {
                int l=0,r=j;
                while (l<=r){
                    int mid=l+(r-l)/2;
                    if (obstacles[i]<dp[mid]){
                        r=mid-1;
                    }else {
                        l=mid+1;//target>=dp[i]
                    }
                }
                dp[l]=obstacles[i];
                ans[i]=l+1;
            }
        }
        return ans;
    }

    public int longestCommonSubsequence(String text1, String text2) {
        int m=text1.length(),n=text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i-1)==text2.charAt(j-1)){
                    dp[i][j]=dp[i-1][j-1]+1;
                }else {
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[m][n];
    }
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int m=nums1.length,n=nums2.length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (nums1[i-1]==nums2[j-1]){
                    dp[i][j]=dp[i-1][j-1]+1;
                }else {
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[m][n];
    }
    public int maxProfit(int[] prices) {
        int n=prices.length;
        int[][] dp = new int[n + 1][2];
        dp[0][0]=0;
        dp[0][1]=-prices[0];
        for (int i = 1; i < n; i++) {
            dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
            dp[i][1]=Math.max(dp[i-1][1],(i-2>=0?dp[i-2][0]:0)-prices[i]);
        }
        return dp[n-1][0];
    }
    public int maxProfit2(int[] prices) {
        int n=prices.length;
        int[][][] dp = new int[n + 1][2][2];
        dp[0][1][0]=0;
        dp[0][1][1]=-prices[0];
        for (int i = 1; i < n; i++) {
            dp[i][1][0]=Math.max(dp[i-1][1][0],dp[i-1][1][1]+prices[i]);
            dp[i][1][1]=Math.max(dp[i-1][1][1],-prices[i]);
        }
        return dp[n-1][1][0];
    }
    public int maxProfit3(int k, int[] prices) {
        int n=prices.length;
        if (k<=n/2){//和k=2的差不多
            int[][][] dp = new int[n + 1][k+1][2];
            dp[0][1][0]=0;
            dp[0][1][1]=-prices[0];
            for (int i = 1; i <= k ; i++) {
                dp[0][i][0]=0;
                dp[0][i][1]=-prices[0];
            }

            for (int i = 1; i < n; i++) {
                for (int j = 1; j <= k; j++) {
                    dp[i][j][0]=Math.max(dp[i-1][j][0],dp[i-1][j][1]+prices[i]);
                    dp[i][j][1]=Math.max(dp[i-1][j][1],dp[i-1][j-1][0]-prices[i]);
                }
            }
            return dp[n-1][k][0];
        }else {//超过临界值和k=无穷大的情况一样
            int[][] dp = new int[n + 1][2];
            dp[0][0]=0;
            dp[0][1]=-prices[0];
            for (int i = 1; i < n; i++) {
                dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
                dp[i][1]=Math.max(dp[i-1][1],dp[i-1][0]-prices[i]);
            }
            return dp[n-1][0];
        }
    }
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0]=1;
        dp[1]=1;
        for (int i = 2; i <=n ; i++) {
            for (int j = 0; j <= i-1 ; j++) {
                dp[i]+=dp[j]*dp[i-j-1];
            }
        }
        return dp[n];
    }
    public List<TreeNode> generateTrees(int n) {
        if (n==0){
            return new LinkedList<TreeNode>();
        }
        return generateTrees(1,n);
    }
    public List<TreeNode> generateTrees(int start,int end) {
        LinkedList<TreeNode> allTrees = new LinkedList<>();
        if (start>end){
            return allTrees;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftTrees = generateTrees(start, i - 1);
            List<TreeNode> rightTrees = generateTrees(i + 1, end);
            for (TreeNode leftTreeNode:leftTrees) {
                for (TreeNode rightTreeNode:rightTrees) {
                    TreeNode treeNode = new TreeNode(i,leftTreeNode,rightTreeNode);
                    allTrees.add(treeNode);
                }
            }
        }
        return allTrees;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        ArrayList<List<Integer>> reList = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i]>0){//如果num-i大于0,num-i和后边的数相加不可能等于0
                break;
            }
            if (i>0&&nums[i]==nums[i-1]){//防止结果重复
                continue;
            }
            int third=nums.length-1;//第三个数初始坐标
            for (int second = i+1; second < third; ) {
                if (second>i+1&&nums[second]==nums[second-1]){//去掉第二个元素的重复数
                    second++;
                    continue;
                }
                if (third<nums.length-1&&nums[third]==nums[third+1]){
                    while (third<nums.length-1&&nums[third]==nums[third+1]){//第三个元素去重
                        third--;
                    }
                    continue;
                }

                if (nums[second]+nums[third]+nums[i]==0){//三数之和等于0
                    ArrayList<Integer> objects = new ArrayList<>();
                    objects.add(nums[i]);
                    objects.add(nums[second]);
                    objects.add(nums[third]);
                    reList.add(objects);
                    third--;
                    second++;
                }
                if (nums[second]+nums[third]>-nums[i]){//第二个和第三个数之和大于第一个，说明第三个太大，如果小于第一个，说明第二个太小；
                    third--;
                }
                if (nums[second]+nums[third]<-nums[i]){//第二个和第三个数之和大于第一个，说明第三个太大，如果小于第一个，说明第二个太小；
                    second++;
                }
            }
        }
        return reList;
    }

    public static void main(String[] args) {
        SnakeGame snakeGame = new SnakeGame();
        snakeGame.threeSum(new int[]{-2,0,0,2,2});


    }
}

  //Definition for a binary tree node.
  class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode() {}
      TreeNode(int val) { this.val = val; }
      TreeNode(int val, TreeNode left, TreeNode right) {
          this.val = val;
          this.left = left;
          this.right = right;
      }
  }

  class ListNode {
      int val;
      ListNode next;
      ListNode() {}
      ListNode(int val) { this.val = val; }
      ListNode(int val, ListNode next) { this.val = val; this.next = next; }
  }
