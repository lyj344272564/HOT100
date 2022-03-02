# LC  HOT100（Java）
高频HOT100（LC）

---

### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

````java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int i=0; i<nums.length; i++) {
            if (map.containsKey(target-nums[i])) {
                return new int[]{i,map.get(target-nums[i])};
            }
            map.put(nums[i],i);
        }
        return new int[] {};
    }
}
````

### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

````java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int a = 0;// 看是否有进位
        ListNode head = null;
        ListNode tail = null;
        while (l1 != null || l2 != null) {
            int n1 = l1!=null?l1.val:0;
            int n2 = l2!=null?l2.val:0;
            int sum = n1 + n2 + a;
            if (null == head) {
                head = tail = new ListNode(sum%10);
            } else {
                tail.next = new ListNode(sum%10);
                tail = tail.next;
            }
            a = sum / 10;
            if (null != l1) l1 = l1.next;
            if (null != l2) l2 = l2.next;
        }
        if (a > 0) {
                tail.next = new ListNode(a);
        }
        return head;

    }
}
````

### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

````java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int len = 0;
        for (int i=0,j=0; j<s.length(); j++) {
            char c = s.charAt(j);
            if (map.containsKey(c)) {
               i = Math.max(i,map.get(c)+1);
            }
            len = Math.max(len,j-i+1);
            map.put(c,j);
        }
        return len;
    }
}
````

### [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

````java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int m = nums2.length;
        int[] temp = new int[n+m];
        int i = 0;
        int j = 0;
        int k = 0;
        while (i<n && j<m) {
            if (nums1[i] <= nums2[j]) {
                temp[k++] = nums1[i++];
            } else {
                temp[k++] = nums2[j++];
            }
        }
        while (i<n) {
            temp[k++] = nums1[i++];
        }
        while (j<m) {
            temp[k++] = nums2[j++];
        }
        int mid = (n+m)/2;
        if ((n+m)%2 == 0) {
            return (double) (temp[mid-1] + temp[mid])/2.0;
        } else {
            return (double) temp[mid];
        }
    }
}
````

### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

````java
class Solution {
    public String longestPalindrome(String s) {
        String res = "";
        for (int i=0; i<s.length(); i++) {
            int l = i-1;
            int r = i+1;
            while (l>=0 && r<s.length() && s.charAt(l)==s.charAt(r)) {
                l--;
                r++;
            }
            if (res.length() < r-l-1) {
                res = s.substring(l+1,r);
            }
             l = i;
             r = i+1;
            while (l>=0 && r<s.length() && s.charAt(l)==s.charAt(r)) {
                l--;
                r++;
            }
            if (res.length() < r-l-1) {
                res = s.substring(l+1,r);
            }
        }
        return res;
    }
}
````

### [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

````java
class Solution {
    public boolean isMatch(String s, String p) {
        int n = s.length();
        int m = p.length();
        boolean[][] dp = new boolean[n+1][m+1];
        for (int i=0; i<=n; i++) {
            for (int j=0; j<=m; j++) {
                if (j==0) {
                    dp[i][j] = i==0;
                } else {
                    if (p.charAt(j-1)!='*') { // 非*
                        if (i>0 && (s.charAt(i-1)==p.charAt(j-1) || p.charAt(j-1)=='.')) {
                            dp[i][j] = dp[i-1][j-1];
                        }
                    } else {// *
                        if (j>=2)  {
                            dp[i][j] |= dp[i][j-2];  // 不看
                        }
                        if (i>=1 && j>=2 && (s.charAt(i-1)==p.charAt(j-2) || p.charAt(j-2)=='.')) {// 看
                            dp[i][j] |= dp[i-1][j];
                        }
                    }
                }
            }
        }
        return dp[n][m];
    }
}
````

### [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

````java
class Solution {
    public int maxArea(int[] height) {
        int res = 0;
        int l = 0;
        int r = height.length-1;
        while (l < r) {
            res = Math.max(res,Math.min(height[l],height[r])*(r-l));
            if (height[l] < height[r]) {
                l++;
            } else {
                r--;
            }
        }
        return res;
    }
}
````

### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

````java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int first=0; first<nums.length; first++) {
            if (first>0 && nums[first] == nums[first-1]) { // 判断和上一次是否相同
                continue;
            }
            int thrid = n - 1; // 第三个数
            for (int second=first+1; second<thrid; second++) {
                if (second>first+1 && nums[second]==nums[second-1]) {
                    continue;
                }
                while (second<thrid && nums[second]+nums[first]+nums[thrid]>0) {
                    thrid--;
                }
                if (second == thrid) break; // 如果此时相等那么之后就不会出现a+b+c==0的情况
                if (nums[second] + nums[first] + nums[thrid] == 0) {
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[thrid]);
                    res.add(list);
                }
            }
        }
        return res;
    }
}
````

### [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

````java
class Solution {
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (0 == digits.length()) return res;
        Map<Character,String> phoneMap = new HashMap<>(){{
            put('2',"abc");
            put('3',"def");
            put('4',"ghi");
            put('5',"jkl");
            put('6',"mno");
            put('7',"pqrs");
            put('8',"tuv");
            put('9',"wxyz");
        }};
        backtrack(res, phoneMap, digits, 0, new StringBuilder());
        return res;
    }
    public void backtrack(List<String> res, Map<Character,String> phoneMap, String digits, int index, StringBuilder sb) {
        if (index == digits.length()) {
            res.add(sb.toString());
        } else {
            char digit = digits.charAt(index);
            String letters = phoneMap.get(digit);
            int lettersLength = letters.length();
            for (int i=0; i<lettersLength; i++) {
                sb.append(letters.charAt(i));
                backtrack(res,phoneMap,digits,index+1,sb);
                sb.deleteCharAt(index);
            }
        }
    }
}
````

### [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

````java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode cur = head;
        while (cur != null) {
            n--;
            cur = cur.next;
        }
        if (n==0) {
            return head.next;
        }
        if (n<0) {
            cur = head;
            while (++n != 0) {
                cur = cur.next;
            }
            cur.next = cur.next.next;
        }
        return head;
    }
}
````

### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

````java
class Solution {
    public boolean isValid(String s) {
        int n = s.length();
        if (n%2 == 1) return false;
        Map<Character,Character> map = new HashMap<>(){{
            put(')','(');
            put(']','[');
            put('}','{');
        }};
        Stack<Character> stack = new Stack<>();
        for (int i=0; i<n; i++) {
            char c = s.charAt(i);
            if (map.containsKey(c)) {
                if (stack.isEmpty() || stack.peek() != map.get(c)) {
                    return false;
                }
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }
}
````

### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

````java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode cur = new ListNode(-1);
        ListNode dum = cur;
        while (list1!=null && list2!=null) {
            if (list1.val <= list2.val) {
                cur.next = list1;
                list1 = list1.next;
            } else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }
        cur.next = list1==null?list2:list1;
        return dum.next;
    }
}
````

### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

````java
class Solution {
    List<String> res = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        dfs(n,0,0,"");
        return res;
    }
    public void dfs(int n, int lc, int rc, String seq) {
        if (lc==n && rc==n) {
            res.add(seq);
        } else {
            if (lc<n) {
                dfs(n,lc+1,rc,seq+'(');
            }
            if (rc<n && lc>rc) {
                dfs(n,lc,rc+1,seq+')');
            }
        }
    }
}
````

### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

````java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        return mergeListNode(lists,0,lists.length-1);
    }

    public ListNode mergeListNode(ListNode[] lists, int l, int r) {
        if (l == r) return lists[l];
        if (l>r) return null;
        int mid = l + r >> 1;
        return merge(mergeListNode(lists,l,mid),mergeListNode(lists,mid+1,r));
    }

    public ListNode merge(ListNode h1, ListNode h2) {
        if (h1 == null || h2 == null) return h1==null?h2:h1;
        ListNode head = new ListNode(0);
        ListNode tail = head;
        ListNode apart = h1;
        ListNode bpart = h2;
        while (apart!=null && bpart!=null) {
            if (apart.val <= bpart.val) {
                tail.next = apart;
                apart = apart.next;
            } else {
                tail.next = bpart;
                bpart = bpart.next;
            }
            tail = tail.next;
        }
        tail.next = apart==null?bpart:apart;
        return head.next;
    }
}
````

### [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

````java
class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i>=0 && nums[i]>=nums[i+1]) {
            i--;
        }
        if (i>=0) {
            int j = nums.length-1;
            while (j>=0 && nums[i]>=nums[j]) {
                j--;
            }
            swap(nums,i,j);
        }
        reverse(nums,i+1);
    }
    public void swap(int[] nums, int a, int b) {
        int tmp = nums[a];
        nums[a] = nums[b];
        nums[b] = tmp;
    }
    public void reverse(int nums[], int start) {
        int l = start;
        int r = nums.length-1;
        while (l < r) {
            swap(nums,l,r);
            l++;
            r--;
        }
    }
}
````

### [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

````java
class Solution {
    public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        int res = 0;
        for (int i=0,start=-1; i<s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty()) {
                    stack.pop();
                    if (!stack.isEmpty()) {
                        res = Math.max(res,i-stack.peek());
                    } else {
                        res = Math.max(res,i-start);
                    }
                } else {
                    start = i;
                }
            }
        }
        return res;
    }
}
````

### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

````java
class Solution {
    public int search(int[] nums, int target) {
        int n = nums.length;
        if (0 == n) return -1;
        int l = 0;
        int r = n - 1;
        while (l<r) {
            int mid = l + r + 1 >> 1;
            if (nums[mid] >= nums[0]) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        if (nums[0] <= target) {
            l = 0;
        } else {
            l = r + 1;
            r = n - 1;
        }
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] >= target) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        if (nums[r] == target) {
            return r;
        }
        return -1;
    }
}
````

### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

````java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int n = nums.length;
        if (0 == n) return new int[]{-1,-1};
        int l = 0;
        int r = n-1;
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] >= target) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        if (nums[l] != target) {
            return new int[]{-1,-1};
        }
        int L = l;
        l = 0;
        r = n - 1;
        while (l < r) {
            int mid = l + r + 1 >> 1;
            if (nums[mid] <= target) {
                l = mid;
            } else {
                r = mid - 1;
            }
        }
        return new int[]{L,l};
    }
}
````

### [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

````java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        dfs(candidates,0,target);
        return res;
    }
    public void dfs (int[] c, int u, int target) {
        if (0 == target) {
            res.add(new ArrayList<>(path));
            return;
        }
        if (u == c.length) return;
        for (int i=0; c[u]*i<=target; i++) {
            dfs(c,u+1,target-c[u]*i);
            path.add(c[u]);
        }
        for (int i=0; c[u]*i<=target; i++) {
            path.remove(path.size()-1);
        }
    }
}
````

### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

````java
class Solution {
    public int trap(int[] height) {
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i=0; i<height.length; i++) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int top = stack.pop();
                if (stack.isEmpty()) break;
                int left = stack.peek();
                int currWidth = i-left-1;
                int currHeight = Math.min(height[i],height[left])-height[top];
                res += currWidth*currHeight;
            }
            stack.push(i);
        }
        return res;
    }
}
````

### [46. 全排列](https://leetcode-cn.com/problems/permutations/)

````java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> tmp = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        dfs(0,nums);
        return res;
    }
    public void dfs(int u, int[] nums) {
        if (u == nums.length) {
            res.add(new ArrayList<>(tmp));
            return;
        }
        for (int i=0; i<nums.length; i++) {
            if (tmp.contains(nums[i])) {
                continue;
            }
            tmp.add(nums[i]);
            dfs(u+1,nums);
            tmp.remove(tmp.size()-1);
        }
    }
}
````

### [48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

````java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int[][] matrix_new = new int[n][n];
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                matrix_new[j][n-i-1] = matrix[i][j];
            }
        }
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                matrix[i][j] = matrix_new[i][j];
            }
        }
    }
}	
````

### [49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

````java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String,List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] c = s.toCharArray();
            Arrays.sort(c);
            String key = String.valueOf(c);
            List<String> list = map.getOrDefault(key,new ArrayList<>());
            list.add(s);
            map.put(key,list);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
````

### [53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

````java
class Solution {
    public int maxSubArray(int[] nums) {
        int s = 0;
        int res = Integer.MIN_VALUE;
        for (int x : nums) {
            if (s<0) {
                s = 0;
            }
            s+=x;
            res = Math.max(res, s);
        }
        return res;
    }
}
````

### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

````java
class Solution {
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int rightmost = 0;
        for (int i=0; i<n; i++) {
            if (i<=rightmost) {
                rightmost = Math.max(rightmost,nums[i]+i);
                if (rightmost>=n-1) {
                    return true;
                }
            }
        }
        return false;
    }
}
````

### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

````java
class Solution {
    public int[][] merge(int[][] a) {
        Arrays.sort(a,(o1,o2)->{return o1[0]-o2[0];});
        LinkedList<int[]> stack = new LinkedList<>();
        for (int arr[] : a) {
            if (!stack.isEmpty() && arr[0] <= stack.peekLast()[1]) {
                stack.peekLast()[1] = Math.max(arr[1], stack.peekLast()[1]);
            } else {
                stack.addLast(arr);
            }
        }
        return stack.toArray(new int[stack.size()][2]);
    }
}
````

### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

````java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i=0; i<n; i++) {
            dp[0][i] = 1;
        }
        for (int i=0; i<m; i++) {
            dp[i][0] = 1;
        }
        for (int i=1; i<m; i++) {
            for (int j=1; j<n; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
}
````

### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

````java
class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = grid[i][j];
                } else if (i == 0) {
                    dp[i][j] = dp[i][j-1] + grid[i][j];
                } else if (j == 0) {
                    dp[i][j] = dp[i-1][j] + grid[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i-1][j],dp[i][j-1]) + grid[i][j];
                }
            }
        }
        return dp[m-1][n-1];
    }
}
````

### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

````java
class Solution {
    public int climbStairs(int n) {
        int p = 0;
        int q = 0;
        int r = 1;
        for (int i=1; i<=n; i++) {
            p = q;
            q = r;
            r = p+q;
        }
        return r;
    }
}
````

### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

````java
class Solution {
    public int minDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        char[] arr1 = word1.toCharArray();
        char[] arr2 = word2.toCharArray();
        int[][] dp = new int[n+1][m+1];
        for (int i=1; i<=n; i++) {
            dp[i][0] = i;
        }
        for (int i=1; i<=m; i++) {
            dp[0][i] = i;
        }
        for (int i=1; i<=n; i++) {
            for (int j=1; j<=m; j++) {
                if (arr1[i-1] == arr2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    int insert = dp[i-1][j];
                    int delete = dp[i][j-1];
                    int replace = dp[i-1][j-1];
                    dp[i][j] = Math.min(insert,Math.min(delete,replace)) + 1;
                }
            }
        }
        return dp[n][m];
    }
}
````

### [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)

````java
class Solution {
    public void sortColors(int[] nums) {
        for (int i=0,j=0,k=nums.length-1; i<=k;) {
            if (nums[i] == 0) {
                swap(nums,i++,j++);
            } else if (nums[i] == 2) {
                swap(nums,i,k--);
            } else {
                i++;
            }
        }
    }
    public void swap(int[] nums, int a, int b) {
        int tmp = nums[a];
        nums[a] = nums[b];
        nums[b] = tmp;
    }
}
````

### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

````java
class Solution {
    Map<Character,Integer> ori = new HashMap<>();
    Map<Character,Integer> cnt = new HashMap<>();
    public String minWindow(String s, String t) {
        int tLen = t.length();
        int sLen = s.length();
        for (int i=0; i<tLen; i++) {
            ori.put(t.charAt(i), ori.getOrDefault(t.charAt(i),0)+1);
        }
        int l = 0;
        int r = -1;
        int len = Integer.MAX_VALUE;
        int ansL = -1;
        int ansR = -1;
        while (r < sLen) {
            ++r;
            if (r<sLen && ori.containsKey(s.charAt(r))) {
                cnt.put(s.charAt(r), cnt.getOrDefault(s.charAt(r),0)+1);
            }
            while (check() && l <= r) {
                if (r-l+1 < len) {
                    len = r-l+1;
                    ansL = l;
                    ansR = l+len;
                }
                if (ori.containsKey(s.charAt(l))) {
                    cnt.put(s.charAt(l), cnt.getOrDefault(s.charAt(l),0)-1);
                }
                ++l;
            }
        }
        return ansL==-1?"":s.substring(ansL,ansR);
    }
    public boolean check() {
        Iterator iter = ori.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry entry = (Map.Entry) iter.next();
            Character key = (Character) entry.getKey();
            Integer value = (Integer) entry.getValue();
            if (cnt.getOrDefault(key,0) < value) {
                return false;
            }
        }
        return true;
    }
}
````

### [78. 子集](https://leetcode-cn.com/problems/subsets/)

````java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        for (int i=0; i<(1<<n); i++) {
            List<Integer> path = new ArrayList<>();
            for (int j=0; j<n; j++) {
                if (((i>>j)&1)!=0) {
                    path.add(nums[j]);
                }
            }
            res.add(new ArrayList<>(path));
        }
        return res;
    }
}
````

### [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

````java
class Solution {
    public boolean exist(char[][] board, String word) {
        for (int i=0; i<board.length; i++) {
            for (int j=0; j<board[0].length; j++) {
                if (dfs(board,word,0,i,j)) {
                    return true;
                }
            }
        }
        return false;
    }
    public boolean dfs(char[][] board, String word, int u, int x, int y) {
        if (x<0 || x>board.length || y<0 || y>board[0].length || board[x][y]!=word.charAt(u)) {
            return false;
        }
        if (u == word.length()-1) return true;
        int[] dx = {0,0,1,-1};
        int[] dy = {1,-1,0,0};
        char c = board[x][y];
        board[x][y] = '*';
        for (int i=0; i<4; i++) {
            int a = x + dx[i];
            int b = y + dy[i];
            if (a>=0&&a<board.length&&b>=0&&b<board[0].length&&board[a][b]!='*') {
                if (dfs(board,word,u+1,a,b)) {
                    return true;
                }
            }
        }
        board[x][y] = c;
        return false;
    }
}
````

### [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

````java
class Solution {
    public int largestRectangleArea(int[] h) {
        int n = h.length;
        int[] left = new int[n];
        int[] right = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i=0; i<n; i++) {
            while (!stack.isEmpty() && h[stack.peek()] >= h[i]) {
                stack.pop();
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        stack.clear();
        for (int i=n-1; i>=0; i--) {
            while (!stack.isEmpty() && h[stack.peek()] >= h[i]) {
                stack.pop();
            }
            right[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
        int res = 0;
        for (int i=0; i<n; i++) {
            res = Math.max(res,h[i]*(right[i]-left[i]-1));
        }
        return res;
    }
}
````

### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

````java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (0 == matrix.length || 0==matrix[0].length) return 0;
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] h = new int[m][n];
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                if (matrix[i][j] == '1') {
                    if (i>0) {
                        h[i][j] = h[i-1][j] + 1;
                    } else {
                        h[i][j] = 1;
                    }
                }
            }
        }
        int res = 0;
        for (int i=0; i<m; i++) {
            res = Math.max(res,largestRectangleArea(h[i]));
        }
        return res;
    }
    public int largestRectangleArea(int[] h) {
        int n = h.length;
        int[] left = new int[n];
        int[] right = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i=0; i<n; i++) {
            while (!stack.isEmpty() && h[stack.peek()] >= h[i]) {
                stack.pop();
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        stack.clear();
        for (int i=n-1; i>=0; i--) {
            while (!stack.isEmpty() && h[stack.peek()] >= h[i]) {
                stack.pop();
            }
            right[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
        int res = 0;
        for (int i=0; i<n; i++) {
            res = Math.max(res,h[i]*(right[i]-left[i]-1));
        }
        return res;
    }
}
````

### [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<Integer> res = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        mergeTrans(root);
        return res;
    }
    public void mergeTrans(TreeNode root) {
        if (null == root) return;
        mergeTrans(root.left);
        res.add(root.val);
        mergeTrans(root.right);
    }
}
````

### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

````java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i=2; i<=n; i++) {
            for (int j=1; j<=i; j++) {
                dp[i] += dp[j-1] * dp[i-j];
            }
        }
        return dp[n];
    }
}
````

### [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean isValidBST(TreeNode root) {
        return dfs(root,Long.MIN_VALUE,Long.MAX_VALUE);
    }
    public boolean dfs(TreeNode root, long lower, long upper) {
        if (null == root) return true;
        if (root.val <= lower || root.val >= upper) return false;
        return dfs(root.left,lower,root.val) && dfs(root.right,root.val,upper);
    }
}
````

### [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (null == root) return true;
        return dfs(root.left,root.right);
    }
    public boolean dfs(TreeNode a, TreeNode b) {
        if (a==null || null==b) return a==null&&b==null;
        if (a.val!=b.val) return false;
        return dfs(a.left,b.right) && dfs(a.right,b.left);
    }
}
````

### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (null == root) return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i=0; i<size; i++) {
                TreeNode t = q.poll();
                tmp.add(t.val);
                if (t.left!=null) q.offer(t.left);
                if (t.right!=null) q.offer(t.right);
            }
            res.add(new ArrayList<>(tmp));
        }
        return res;
    }
}
````

### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        if (null == root) {
            return 0;
        } else {
            int leftHeight = maxDepth(root.left);
            int rightHeight = maxDepth(root.right);
            return Math.max(leftHeight,rightHeight) + 1;
        }
    }
}
````

### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    Map<Integer, Integer> map = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for (int i=0; i<inorder.length; i++) {
            map.put(inorder[i],i);
        }
        return builderTree(preorder,0,inorder,0,inorder.length-1);
    }

    public TreeNode builderTree(int[] preorder, int preIndex, int[] inorder, int inStart, int inEnd) {
        if (inStart>inEnd) return null;
        int val = preorder[preIndex];
        int inIndex = map.get(val);
        TreeNode root = new TreeNode(val);
        int leftNum = inIndex - inStart;
        root.left = builderTree(preorder, preIndex+1,inorder,inStart,inIndex-1);
        root.right = builderTree(preorder,preIndex+1+leftNum,inorder,inIndex+1,inEnd);
        return root;
    }
}
````

### [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public void flatten(TreeNode root) {
        List<TreeNode> list = new ArrayList<>();
        preorderTraversal(root, list);
        int size = list.size();
        for (int i=1; i<size; i++) {
            // TreeNode prev = list.get(i-1);
            TreeNode prev = list.get(i-1);
            TreeNode cur = list.get(i);
            prev.left = null;
            prev.right = cur;
        }
    }
    public void preorderTraversal(TreeNode root, List<TreeNode> list) {
        if (null == root) return;
        list.add(root);
        preorderTraversal(root.left,list);
        preorderTraversal(root.right,list);
    }
}
````

### [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

````java
class Solution {
    public int maxProfit(int[] prices) {
        int res = 0;
        for (int i=0,min=Integer.MAX_VALUE; i<prices.length; i++) {
            res = Math.max(res,prices[i]-min);
            min = Math.min(min,prices[i]);
        }
        return res;
    }
}
````

### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int res = 0;
    public int maxPathSum(TreeNode root) {
        res = Integer.MIN_VALUE;
        dfs(root);
        return res;
    }
    public int dfs(TreeNode root) {
        if (null == root) return 0;
        int leftNum = Math.max(0,dfs(root.left));
        int rightNum = Math.max(0,dfs(root.right));
        res = Math.max(res, root.val + leftNum + rightNum);
        return root.val + Math.max(leftNum,rightNum);
    }
}
````

### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

````java
class Solution {
    public int longestConsecutive(int[] nums) {
        HashSet<Integer> s = new HashSet<>();
        for (int x : nums) {
            s.add(x);
        }
        int res = 0;
        for (int x : nums) {
            if (s.contains(x) && !s.contains(x-1)) {
                int y = x;
                s.remove(x);
                while (s.contains(y+1)) {
                    y++;
                    s.remove(y);
                }
                res = Math.max(res,y-x+1);
            }
          
        }
        return res;
    }
}
````

### [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

````java
class Solution {
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int x : nums) {
            res ^= x;
        }
        return res;
    }
}
````

### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

````java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length()+1];
        dp[0] = true;
        for (int i=1; i<=s.length(); i++) {
            for (int j=0; j<i; j++) {
                if (dp[j] && set.contains(s.substring(j,i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}
````

### [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

````java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        ListNode s = head;
        ListNode f = head.next;
        while (f != null) {
            s = s.next;
            f = f.next;
            if (f == null) return false;
            f= f.next;
            if (s == f) {
                return true;
            }
        }
        return false;
    }   
}
````

### [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

````java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) return null;
        ListNode s = head;
        ListNode f = head.next;
        while (f != null) {
            s = s.next;
            f = f.next;
            if (f == null) return null;
            f = f.next;
            if (s == f) {
                s = head;
                f = f.next;
                while (s != f) {
                    s = s.next;
                    f = f.next;
                }
                return f;
            }
        }
        return null;
    }
}
````

### [146. LRU 缓存](https://leetcode-cn.com/problems/lru-cache/)

````java
class LRUCache {
    class Node {
        int key;
        int value;
        Node pre;
        Node next;
        public Node(){}
        public Node(int _key, int _value) {
            key = _key;
            value = _value;
        }
    }
    private Map<Integer, Node> cache = new HashMap<>();
    private int size;
    private int capacity;
    private Node tail, head;
    public LRUCache(int capacity) {
        // this.size = size;
        this.size = 0;
        this.capacity = capacity;
        tail = new Node();
        head = new Node();
        head.next = tail;
        tail.pre = head;
    }
    
    public int get(int key) {
        Node node = cache.get(key);
        if (null == node) {return -1;}
        moveHead(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        Node node = cache.get(key);
        if (null == node) {
            Node newNode = new Node(key,value);
            cache.put(key,newNode);
            addToHead(newNode);
            ++size;
            if (size > capacity) {
                Node tail = removeTail();
                cache.remove(tail.key);
                --size;
            }
        } else {
            node.value = value;
            moveHead(node);
        }
    }
    
    private void addToHead(Node node) {
        node.pre = head;
        node.next = head.next;
        head.next.pre = node;
        head.next = node;
    }
    private void removeNode(Node node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
    }
    private void moveHead(Node node) {
        removeNode(node);
        addToHead(node);
    }
    private Node removeTail() {
        Node res = tail.pre;
        removeNode(res);
        return res;
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
````

### [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

````java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        if (null == head) return head;
        int n = 0;
        ListNode node = head;
        while (node != null) {
            n++;
            node = node.next;
        }
        ListNode dum = new ListNode(0, head);
        for (int subLength=1; subLength<n; subLength<<=1) {
            ListNode pre = dum;
            ListNode cur = dum.next;
            while (cur != null) {
                ListNode h1 = cur;
                for (int i=1; i<subLength&&cur.next!=null; i++) {
                    cur = cur.next;
                }
                ListNode h2 = cur.next;
                cur.next = null;
                cur = h2;
                for (int i=1; i<subLength&&cur!=null&&cur.next!=null; i++) {
                    cur = cur.next;
                }
                ListNode next = null;
                if (cur != null) {
                    next = cur.next;
                    cur.next = null;
                }
                ListNode merged = merge(h1,h2);
                pre.next = merged;
                while (pre.next!=null) {
                    pre = pre.next;
                }
                cur = next;
            }
        }
        return dum.next;
    }
    public ListNode merge(ListNode h1, ListNode h2) {
        ListNode dum = new ListNode(0);
        ListNode temp = dum;
        ListNode t1 = h1, t2 = h2;
        while (t1!=null && t2!=null) {
            if (t1.val <= t2.val) {
                temp.next = t1;
                t1 = t1.next;
            } else {
                temp.next = t2;
                t2 = t2.next;
            }
            temp = temp.next;
        }
        temp.next = t1==null ? t2 : t1;
        return dum.next;
    }
}
````

### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

````java
class Solution {
    public int maxProduct(int[] nums) {
        int len = nums.length;
        int[] maxF = new int[len];
        int[] minF = new int[len];
        System.arraycopy(nums,0,maxF,0,len);
        System.arraycopy(nums,0,minF,0,len);
        for (int i=1; i<len; i++) {
            maxF[i] = Math.max(nums[i]*maxF[i-1], Math.max(nums[i],minF[i-1]*nums[i]));
            minF[i] = Math.min(nums[i]*minF[i-1], Math.min(nums[i],maxF[i-1]*nums[i]));
        }
        int ans = maxF[0];
        for (int i=1; i<len; i++) {
            ans = Math.max(ans,maxF[i]);
        }
        return ans;
    }
}
````

### [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

````java
class MinStack {

    Stack<Integer> stackData = new Stack<>();
    Stack<Integer> stackMin = new Stack<>();
 
    public MinStack() {

    }
    
    public void push(int val) {
        stackData.push(val);
        if (stackMin.isEmpty() || val <= stackMin.peek()) {
            stackMin.push(val);
        }
    }
    
    public void pop() {
        if (stackData.pop().equals(stackMin.peek())) {
            stackMin.pop();
        }
    }
    
    public int top() {  
        return stackData.peek();
    }
    
    public int getMin() {
        return stackMin.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(val);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
````

### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

````java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (null == headA || null == headB) return null;
        ListNode h1 = headA;
        ListNode h2 = headB;
        while (h1 != h2) {
            h1 = h1==null? headB : h1.next;
            h2 = h2==null? headA : h2.next;
        }
        return h1;
    }
}
````

### [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

````java
class Solution {
    public int majorityElement(int[] nums) {
        int r = 0;
        int c = 0;
        for (int x : nums) {
            if (c == 0) {
                r = x;
                c++;
            } else if (r == x) {
                c++;
            } else {
                c--;
            }
        }
        return r;
    }
}
````

### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

````java
class Solution {
    public int rob(int[] nums) {
       int[] dp = new int[nums.length];
       if (nums.length == 1) {
           return nums[0];
       }
       dp[0] = nums[0];
       dp[1] = Math.max(nums[0], nums[1]);
       for (int i=2; i<nums.length; i++) {
           dp[i] = Math.max(dp[i-2]+nums[i],dp[i-1]);
       }
       return dp[nums.length-1];
    }
}
````

### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

````java
class Solution {
    char[][] g;
    int[] dx = {0,0,1,-1};
    int[] dy = {1,-1,0,0};
    int cnt;
    public int numIslands(char[][] grid) {
        g = grid;
        for (int i=0; i<g.length; i++) {
            for (int j=0; j<g[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(i,j);
                    cnt++;        
                }
            }
        }
        return cnt;
    }

    public void dfs(int x, int y) {
        g[x][y] = '0';
        for (int i=0; i<4; i++) {
            int a = x + dx[i];
            int b = y + dy[i];
            if (a>=0&&a<g.length&&b>=0&&b<g[0].length&&g[a][b]=='1') {
                dfs(a,b);
            }
        }
    }
}
````

### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

````java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
````

### [207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

````java
class Solution {
    public boolean canFinish(int n, int[][] prerequisites) {
        List<List<Integer>> g = new ArrayList<>(); // 领接表
        int[] d = new int[n]; // 入度
        for (int i=0; i<n; i++) {
            g.add(new ArrayList<>());
        }
        for (int[] p : prerequisites) {
            int b = p[0];
            int a = p[1];
            g.get(a).add(b);
            d[b]++;
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i=0; i<n; i++) {
            if (d[i] == 0) {
                q.offer(i);
            }
        }
        int cnt = 0;
        while (q.size() != 0) {
            int t = q.poll();
            cnt++;
            for (int m : g.get(t)) {
                if (--d[m] == 0) {
                    q.offer(m);
                }
            }
        }
        return cnt==n;
    }
}
````

### [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

````java
class Trie {
    private TrieNode root;
    class TrieNode {
        TrieNode[] next;
        boolean end;
        public TrieNode() {
            next = new TrieNode[26];
            end = false;
        }
    }
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (cur.next[idx] == null) {
                cur.next[idx] = new TrieNode();
            }
            cur = cur.next[idx];
        }
        cur.end = true;
    }
    
    public boolean search(String word) {
        TrieNode cur = root;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (cur.next[idx] == null) {
                return false;
            }
            cur = cur.next[idx];
        }
        return cur.end;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode cur = root;
         for (char c : prefix.toCharArray()) {
            int idx = c - 'a';
            if (cur.next[idx] == null) {
                return false;
            }
            cur = cur.next[idx];
        }
        return true;
    }
}

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * boolean param_2 = obj.search(word);
 * boolean param_3 = obj.startsWith(prefix);
 */
````

### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

````java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        quick_sort(nums,0, nums.length-1);
        return nums[nums.length-k];
    }
    public void quick_sort(int[] q, int l, int r) {
        if (l >= r) return;
        int x = q[l];
        int i = l - 1;
        int j = r + 1;
        while (i < j) {
            do {
                i++;
            } while (q[i]<x);
            do {
                j--;
            } while(q[j]>x);
            if (i<j) {
                int tmp = q[i];
                q[i] = q[j];
                q[j] = tmp;
            }
        }
        quick_sort(q,l,j);
        quick_sort(q,j+1,r);
    }
}
````

### [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

````java
class Solution {
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dp = new int[m+1][n+1];
        int res = 0;
        for (int i=1; i<=m; i++) {
            for (int j=1; j<=n; j++) {
                if (matrix[i-1][j-1] == '1') {
                    dp[i][j] = Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1])) + 1;
                    res = Math.max(res,dp[i][j]);
                }
            }
        }
        return res*res;
    }
}
````

### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (null == root) return null;
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }
}
````

### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

````java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next==null) return true;
        Stack<Integer> stack = new Stack<>();
        ListNode cur = head;
        while (cur!=null) {
            stack.push(cur.val);
            cur = cur.next;
        }
        cur = head;
        while (cur != null) {
            if (stack.pop()!=cur.val) {
                return false;
            }
            cur = cur.next;
        }
        return true;
    }
}
````

### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        return help(root,p,q);
    }
    public TreeNode help(TreeNode root, TreeNode p, TreeNode q) {
        if (null == root || p==root || q==root) return root;
        TreeNode l = help(root.left,p,q);
        TreeNode r = help(root.right,p,q);
        if (null == l) return r;
        if (null == r) return l;
        return root;
    }
}
````

### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

````java
class Solution {
    public int[] productExceptSelf(int[] a) {
       int[] b = new int[a.length];
       for (int i=0,n=1; i<a.length; i++) {
            b[i] = n;
            n *= a[i];
       }
       for (int i=a.length-1,n=1; i>=0; i--) {
           b[i] *= n;
           n *= a[i];
       }
       return b;
    }
}
````

### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

````java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
       int[] res = new int[nums.length-k+1];
       if (0==nums.length || k>nums.length) return new int[]{};
       LinkedList<Integer> qmax = new LinkedList<>();
       int cnt = 0;
       for (int i=0; i<nums.length; i++) {
           while (!qmax.isEmpty() && nums[qmax.peekLast()] <= nums[i]) {
               qmax.pollLast();
           }
           qmax.addLast(i);
           if (qmax.peekFirst() == i-k) {
               qmax.pollFirst();
           }
           if (i >= k-1) {
               res[cnt++] = nums[qmax.peekFirst()];
           }
       }
       return res;
    }
}
````

### [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

````java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int i = 0;
        int j = n-1;
        while (i<m && j>=0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                j--;
                
            }
        }
        return false;
    }
}
````

### [253. 会议室 II](https://leetcode-cn.com/problems/meeting-rooms-ii/)

````java
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        PriorityQueue<Integer> allocator = new PriorityQueue<>(intervals.length,
            new Comparator<Integer>(){
                public int compare(Integer a, Integer b) {
                    return a-b;
                }
            }
        );
        Arrays.sort(intervals,
            new Comparator<int[]>() {
                public int compare(int[] a, int[] b) {
                    return a[0]-b[0];
                }
            }
        );
        allocator.add(intervals[0][1]);
        for (int i=1; i<intervals.length; i++) {
            if (intervals[i][0] >= allocator.peek()) {
                allocator.poll();
            }
            allocator.add(intervals[i][1]);
        }
        return allocator.size();
    }
}
````

### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

````java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n+1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i=1; i<=n; i++) {
            for (int j=1; j*j<=i; j++) {
                dp[i] = Math.min(dp[i], dp[i-j*j]+1);
            }
        }
        return dp[n];
    }
}
````

### [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)

````java
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int l = 0; // 指向0
        int r = 0; // 不为0
        while (r < n) {
            if (nums[r] != 0) {
                swap(nums,l,r);
                l++;
            }
            r++;
        }
    }
    public void swap(int[] nums, int l, int r) {
        int tmp = nums[l];
        nums[l] = nums[r];
        nums[r] = tmp;
    }
}
````

### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

````java
class Solution {
    public int findDuplicate(int[] nums) {
        int a = 0;
        int b = 0;
        while (true) {
            a = nums[a];
            b = nums[nums[b]];
            if (a == b) {
                a = 0;
                while (a!=b) {
                    a = nums[a];
                    b = nums[b];
                }
                return a;
            }
        }
    }
}
````

### [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (null == root) return "#";
        StringBuilder sb = new StringBuilder();
        Deque<TreeNode> stack = new LinkedList<>();
        stack.offerLast(root);
        while (!stack.isEmpty()) {
            TreeNode t = stack.pollLast();
            if (null == t) {
                sb.append("#").append(",");
            } else {
                sb.append(t.val).append(",");
                stack.offerLast(t.right);
                stack.offerLast(t.left);
            }
        }
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
        return builderTree(queue);
    }

    public TreeNode builderTree(Deque<String> queue) {
        String s = queue.poll();
        if ("#".equals(s)) {
            return null;
        }
        int val = Integer.parseInt(s);
        TreeNode root = new TreeNode(val);
        root.left = builderTree(queue);
        root.right = builderTree(queue);
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec ser = new Codec();
// Codec deser = new Codec();
// TreeNode ans = deser.deserialize(ser.serialize(root));
````

### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

````java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] q = new int[n+1];
        int len = 0;
        for (int x : nums) {
            int l = 0;
            int r = len;
            while (l < r) {
                int mid = l + r + 1 >> 1;
                if (q[mid] < x) {
                    l = mid;
                } else {
                    r = mid - 1;
                }
            }
            len = Math.max(len, r+1);
            q[r+1] = x;
        }
        return len;
    }
}
````

### [301. 删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)

````java
class Solution {
    List<String> res = new ArrayList<>();
    public List<String> removeInvalidParentheses(String s) {
        int l = 0;
        int r = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                l++;
            } else if (c == ')') {
                if (l == 0) {
                    r++;
                } else {
                    l--;
                }
            }
        }
        dfs(s,0,"",0,l,r);
        return res;
    }
    public void dfs(String s, int u, String path, int cnt, int l, int r) {
        if (u == s.length()) {
            if (0 == cnt) {
                res.add(path.toString());
            }
            return;
        }
        if (s.charAt(u)!='(' && s.charAt(u)!=')') {
            dfs(s,u+1,path+String.valueOf(s.charAt(u)), cnt, l, r);
        } else if (s.charAt(u) == '(') {
            int k = u;
            while (k<s.length() && s.charAt(k)=='(') k++;
            l -= k-u;
            for (int i=k-u; i>=0; i--) {
                if (l>=0) dfs(s,k,path,cnt,l,r);
                path+='(';
                cnt++;
                l++;
            }
        } else if (s.charAt(u) == ')') {
            int k = u;
            while (k<s.length() && s.charAt(k)==')') k++;
            r -= k-u;
            for (int i=k-u; i>=0; i--) {
                if (cnt >= 0 && r>=0) dfs(s,k,path,cnt,l,r);
                path+=')';
                cnt--;
                r++;
            }
        }
    }
}
````

### [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

````java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][3];
        for (int i=0; i<n; i++) {
            Arrays.fill(dp[i],Integer.MIN_VALUE);
        }
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i=1; i<prices.length; i++) {
            dp[i][0] = Math.max(dp[i-1][0],dp[i-1][2]); 
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0]-prices[i]);
            dp[i][2] = dp[i-1][1] + prices[i];
        }
        return Math.max(dp[n-1][0], Math.max(dp[n-1][1], dp[n-1][2]));
    }   
}
````

### [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)

````java
class Solution {
    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[] boll = new int[n+2];
        int[][] dp = new int[n+2][n+2];
        for (int i=0; i<n+2; i++) {
            if (i==0 || i==n+1) {
                boll[i] = 1;
            } else {
                boll[i] = nums[i-1];
            }
        }
        for (int len=3; len<=n+2; len++) {
            for (int i=0; i+len-1<n+2; i++) {
                int j = i+len-1;
                for (int k=i+1; k<j; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][k]+dp[k][j]+boll[i]*boll[k]*boll[j]);
                }
            }
        }
        return dp[0][n+1];
    }
}
````

### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

````java
class Solution {
    public int coinChange(int[] coins, int m) {
        int n = coins.length;
        int[] dp = new int[m+1];
        Arrays.fill(dp,m+1);
        dp[0] = 0;
        for (int u : coins) {
            for (int j=u; j<=m; j++) {
                dp[j] = Math.min(dp[j], dp[j-u]+1);
            }
        }
        if (dp[m]>m) return -1;
        return dp[m]; 
    }
}
````

### [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int rob(TreeNode root) {
        int[] k = dfs(root);
        return Math.max(k[0],k[1]);
    }   
    public int[] dfs(TreeNode root) {
        if (null == root) return new int[]{0,0};
        int[] l = dfs(root.left);
        int[] r=  dfs(root.right);
        return new int[]{Math.max(l[1],l[0])+Math.max(r[0],r[1]), root.val+l[0]+r[0]};
    }
}
````

### [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/)

````java
class Solution {
    public int[] countBits(int n) {
        int[] f = new int[n+1];
        for (int i=0; i<=n; i++) {
            f[i] = cnt(i);
        }
        return f;
    }
    public int cnt(int n) {
        int res = 0;
        while (n!=0) {
            res += n&1;
            n>>=1;
        }
        return res;
    }
}
````

### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

````java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int x : nums) {
            map.put(x,map.getOrDefault(x,0)+1);
        }
        int[] cnt = new int[nums.length+1];
        for (Integer t : map.values()) {
            cnt[t]++;
        }
        int m = 0;
        int n = nums.length;
        while (m != k) {
            m += cnt[n];
            n--;
        }
        int[] res = new int[k];
        int i = 0;
        for (Integer t : map.keySet()) {
            if (map.get(t) > n) {
                res[i++] = t;
            }
        }
        return res;
    }
}
````

### [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)

````java
class Solution {
    int u;
    public String decodeString(String s) {
        StringBuilder res = dfs(s);
        return res.toString();
    }
    public StringBuilder dfs(String s) {
        StringBuilder sb = new StringBuilder();
        while (u<s.length() && s.charAt(u)!=']') {
            if (s.charAt(u)>='a'&&s.charAt(u)<='z' || s.charAt(u)>='A'&&s.charAt(u)<='Z') {
                sb.append(s.charAt(u++));
            } else {
                int t = 0;
                while (s.charAt(u)!='[') t = t*10 + s.charAt(u++)-'0';
                u++; // 过滤右括号
                StringBuilder cur = dfs(s);
                for (int i=0; i<t; i++) {
                    sb.append(cur);
                }
            }
        }
        u++;
        return sb;
    }
}
````

### [406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

````java
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people,new Comparator<int[]>(){
            public int compare(int[] p1, int[] p2) {
                if (p1[0]!=p2[0]){
                    return p2[0]-p1[0];
                } else {
                    return p1[1]-p2[1];
                }
            }
        });
        List<int[]> res = new ArrayList<int[]>();
        for (int[] p : people) {
            res.add(p[1],p);
        }
        return res.toArray(new int[res.size()][]);
    }
}
````

### [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

````java
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int x : nums) {
            sum += x;
        }
        if (sum % 2 == 1) return false;
        sum /= 2;
        boolean[] dp = new boolean[sum+1];
        dp[0] = true;
        for (int x : nums) {
            for (int i=sum; i>=x; i--) {
                dp[i] |= dp[i-x];
            }
        }
        return dp[sum];
    }
}
````

### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int cnt = 0;
    public int pathSum(TreeNode root, int targetSum) {
        if (null == root) return cnt;
        dfs(root,targetSum);
        pathSum(root.left,targetSum);
        pathSum(root.right,targetSum);
        return cnt;
    }

    public void dfs(TreeNode root, int target) {
        if (null == root) return;
        target -= root.val;
        if (target == 0) {
            cnt++;
        }
        dfs(root.right,target);
        dfs(root.left,target);
    }
}
````

### [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

````java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        Map<Character, Integer> cnt = new HashMap<>();
        for (char c : p.toCharArray()) {
            cnt.put(c, cnt.getOrDefault(c,0)+1);
        }
        for (int i=0,j=0,satisfy=0; i<s.length(); i++) {
            if (cnt.containsKey(s.charAt(i))) {
                cnt.put(s.charAt(i), cnt.get(s.charAt(i))-1);
                if (0 == cnt.get(s.charAt(i))) satisfy++;
            }
            while (i-j+1 > p.length()) {
                if (cnt.containsKey(s.charAt(j))) {
                    if (0 == cnt.get(s.charAt(j))) satisfy--;
                    cnt.put(s.charAt(j), cnt.get(s.charAt(j))+1);
                }
                j++;
            }
            if (satisfy == cnt.keySet().size()) res.add(j);
        }
        return res;
    }
}
````

### [448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

````java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        for (int x : nums) {
            x = Math.abs(x);
            if (nums[x-1] > 0) {
                nums[x-1] *= -1;
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i=0; i<nums.length; i++) {
            if (nums[i] > 0) {
                res.add(i+1);
            }
        }
        return res;
    }
}
````

### [461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)

````java
class Solution {
    public int hammingDistance(int x, int y) {
        int res = 0;
        while (x!=0 || y!=0) {
            res += (x&1) ^ (y&1);
            x>>=1;
            y>>=1;
        }
        return res;
    }
}
````

### [494. 目标和](https://leetcode-cn.com/problems/target-sum/)

````java
class Solution {
    public int findTargetSumWays(int[] a, int target) {
        if (target>1000 || target<-1000) return 0;
        int offset = 1000;
        int[][] f = new int[a.length+1][2001];
        f[0][offset] = 1;
        for (int i=1; i<=a.length; i++) {
            for (int j=-1000; j<=1000; j++) {
                if (j-a[i-1] >= -1000) {
                    f[i][j+offset] += f[i-1][j-a[i-1]+offset];
                }
                if (j+a[i-1] <= 1000) {
                    f[i][j+offset] += f[i-1][j+a[i-1]+offset];
                }
            }
        }
        return f[a.length][target+offset];
    }
}
````

### [538. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int sum;
    public TreeNode convertBST(TreeNode root) {
        dfs(root);
        return root;
    }
    public void dfs(TreeNode root) {
        if (null == root) return;
        dfs(root.right);
        int x = root.val;
        root.val += sum;
        sum += x;
        dfs(root.left);
    } 
}
````

### [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int res;
    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return res;
    }
    public int dfs(TreeNode root) {
        if (null == root) return 0;
        int L = dfs(root.left);
        int R = dfs(root.right);
        res = Math.max(res, L+R);
        return Math.max(L,R)+1;
    }
}
````

### [560. 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

````java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int cnt = 0;
        int pre = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0,1);
        for (int i=0; i<nums.length; i++) {
            pre += nums[i];
            if (map.containsKey(pre-k)) {
                cnt += map.get(pre-k);
            }
            map.put(pre,map.getOrDefault(pre,0)+1);
        }
        return cnt;
    }
}
````

### [581. 最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

````java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        if (isSorted(nums)) return 0;
        int[] numsSort = new int[nums.length];
        System.arraycopy(nums, 0, numsSort, 0, nums.length);
        Arrays.sort(numsSort);
        int l = 0;
        int r = nums.length - 1;
        while (nums[l] == numsSort[l]) {
            l++;
        }
        while (nums[r] == numsSort[r]) {
            r--;
        }
        return r - l + 1;
    }
    public boolean isSorted(int[] nums) {
        for (int i=1; i<nums.length; i++) {
            if (nums[i]<nums[i-1]) return false;
        }
        return true;
    }
}
````

### [617. 合并二叉树](https://leetcode-cn.com/problems/merge-two-binary-trees/)

````java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (null == root1) return root2;
        if (null == root2) return root1;
        TreeNode root = new TreeNode(root1.val + root2.val);
        root.left = mergeTrees(root1.left,root2.left);
        root.right = mergeTrees(root1.right, root2.right);
        return root;
    }
}
````

### [621. 任务调度器](https://leetcode-cn.com/problems/task-scheduler/)

````java
class Solution {
    public int leastInterval(char[] tasks, int n) {
        Map<Character,Integer> map = new HashMap<>();
        for (Character c : tasks) {
            map.put(c, map.getOrDefault(c,0)+1);
        }
        int maxc = 0;
        int cnt = 0;
        for (Character c : map.keySet()) {
            maxc = Math.max(maxc, map.get(c));
        }
        for (Character c : map.keySet()) {
            if (maxc == map.get(c)) {
                cnt++;
            }
        }
        return Math.max(tasks.length, (maxc-1)*(n+1)+cnt);
    }
}
````

### [647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

````java
class Solution {
    public int countSubstrings(String s) {
        int res = 0;
        for (int i=0; i<s.length(); i++) {
            for (int j=i,k=i; j>=0&&k<s.length(); j--,k++) {
                if (s.charAt(j)!=s.charAt(k)) break;
                res++;
            }
            for (int j=i,k=i+1; j>=0&&k<s.length(); j--,k++) {
                if (s.charAt(j)!=s.charAt(k)) break;
                    res++;
            }
        }
        return res;
    }
}
````

### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

````java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> stk = new Stack<>();
        int[] res = new int[temperatures.length];
        for (int i=res.length-1; i>=0; i--) {
            while (!stk.isEmpty() && temperatures[i] >= temperatures[stk.peek()]) {
                stk.pop();
            }
            if (!stk.isEmpty()) res[i] = stk.peek()-i;
            stk.push(i);
        }
        return res;
    }
}
````



