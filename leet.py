from typing import List
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        total_prod = 1
        indice_zero = -1
        for index, num in enumerate(nums):
            if num == 0:
                if indice_zero!=-1:
                    indice_zero = index
                    
                else:
                    return [0]*len(nums)
            else:
                total_prod *= num
        print(indice_zero)
        if indice_zero!=-1:
            return_list = [0]*len(nums)
            return_list[indice_zero] = total_prod
            return return_list

        return [
            int(total_prod/num)
            for num in nums
        ]
        
# test this
if __name__ == "__main__":
    s = Solution()
    print(s.productExceptSelf([1,2,3,4]))  # [24,12,8,6]
    print(s.productExceptSelf([0,0]))  # [0,0]
    print(s.productExceptSelf([1,0]))  # [0,1]
    print(s.productExceptSelf([1]))  # [1]
    print(s.productExceptSelf([1,2]))  # [2,1]