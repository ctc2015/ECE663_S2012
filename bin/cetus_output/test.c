#include <stdio.h>
 main()
{
const int size = 1000000000;
int i, sum;
int a[size];
for (i=0; i<size; i ++ )
{
a[i]=i;
}
sum=0;
for (i=0; i<size; i ++ )
{
sum=(sum+a[i]);
}
printf("sum is %d\n", sum);
return 0;
}

