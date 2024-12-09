#include"msm.hpp"
//arithmetic
uint384 montgomery_reduce(uint768 t, 
                          const uint384& m, 
                          const uint384& inv)
{
	uint384 t_low = t(383, 0);
	uint768 temp=karatsuba_384(t_low, inv);
	uint384 temp_low=temp(383,0);
	uint768 x;
	x=t+karatsuba_384(temp_low,m);
	uint384 res=x(767,384);
	if(res>=m)
	{
		res=res-m;
	}
	return res;
}
ap_uint<1536> karatsuba_768(ap_uint<768> x, 
                            ap_uint<768> y)
{
	if (x == 0 || y == 0) return 0;
    ap_uint<384> x_low = x(383, 0);
    ap_uint<384> x_high = x(767, 384);
    ap_uint<384> y_low = y(383, 0);
    ap_uint<384> y_high = y(767, 384);

    ap_uint<385> z1_left = x_low + x_high;
    ap_uint<385> z1_right = y_low + y_high;
    ap_uint<384> low1 = z1_left(383,0);
    ap_uint<384> low2 = z1_right(383,0);
    ap_uint<1536> z0 = karatsuba_384(x_low, y_low);
    ap_uint<1536> z2 = karatsuba_384(x_high, y_high);
    ap_uint<768> z1t = karatsuba_384(low1, low2);
    ap_uint<1536> temp=z1_left[384]*z1_right[384];
    ap_uint<1536> z1 = z1t+(ap_uint<768>(z1_left[384]*low2)<<384)+(ap_uint<768>(z1_right[384]*low1)<<384)+(temp<<768);
    ap_uint<1536> res = (z2 << 768) + ((z1-z0-z2) << 384) + z0;
    return res;
}
ap_uint<768> karatsuba_384(ap_uint<384> x, 
                           ap_uint<384> y)
{
	if (x == 0 || y == 0) return 0;
    ap_uint<192> x_low = x(191, 0);
    ap_uint<192> x_high = x(383, 192);
    ap_uint<192> y_low = y(191, 0);
    ap_uint<192> y_high = y(383, 192);
    ap_uint<193> z1_left = x_low + x_high;
    ap_uint<193> z1_right = y_low + y_high;
    ap_uint<192> low1 = z1_left(191,0);
    ap_uint<192> low2 = z1_right(191,0);
    ap_uint<768> z0 = karatsuba_192(x_low, y_low);
    ap_uint<768> z2 = karatsuba_192(x_high, y_high);
    ap_uint<384> z1t = karatsuba_192(low1, low2);
    ap_uint<768> temp=z1_left[192]*z1_right[192];
    ap_uint<768> z1 = z1t+(ap_uint<384>(z1_left[192]*low2)<<192)+(ap_uint<384>(z1_right[192]*low1)<<192)+(temp<<384);
    ap_uint<768> res = (z2 << 384) + ((z1-z0-z2) << 192) + z0;
    return res;
}
ap_uint<384> karatsuba_192(ap_uint<192> x, 
                           ap_uint<192> y)
{
	if (x == 0 || y == 0) return 0;
    ap_uint<96> x_low = x(95, 0);
    ap_uint<96> x_high = x(191, 96);
    ap_uint<96> y_low = y(95, 0);
    ap_uint<96> y_high = y(191, 96);

    ap_uint<97> z1_left = x_low + x_high;
    ap_uint<97> z1_right = y_low + y_high;
    ap_uint<96> low1 = z1_left(95,0);
    ap_uint<96> low2 = z1_right(95,0);
    ap_uint<384> z0 = karatsuba_96(x_low, y_low);
    ap_uint<384> z2 = karatsuba_96(x_high, y_high);
    ap_uint<192> z1t = karatsuba_96(low1, low2);
    ap_uint<384> temp=z1_left[96]*z1_right[96];
    ap_uint<384> z1 = z1t+(ap_uint<192>(z1_left[96]*low2)<<96)+(ap_uint<192>(z1_right[96]*low1)<<96)+(temp<<192);
    ap_uint<384> res = (z2 << 192) + ((z1-z0-z2) << 96) + z0;
    return res;
}
ap_uint<192> karatsuba_96(ap_uint<96> x, 
                          ap_uint<96> y)
{
	if (x == 0 || y == 0) return 0;
    ap_uint<48> x_low = x(47, 0);
    ap_uint<48> x_high = x(95, 48);
    ap_uint<48> y_low = y(47, 0);
    ap_uint<48> y_high = y(95, 48);

    ap_uint<49> z1_left = x_low + x_high;
    ap_uint<49> z1_right = y_low + y_high;
    ap_uint<48> low1 = z1_left(47,0);
    ap_uint<48> low2 = z1_right(47,0);
    ap_uint<192> z0 = karatsuba_48(x_low, y_low);
    ap_uint<192> z2 = karatsuba_48(x_high, y_high);
    ap_uint<96> z1t = karatsuba_48(low1, low2);
    ap_uint<192> temp=z1_left[48]*z1_right[48];
    ap_uint<192> z1 = z1t+(ap_uint<96>(z1_left[48]*low2)<<48)+(ap_uint<96>(z1_right[48]*low1)<<48)+(temp<<96);
    ap_uint<192> res = (z2 << 96) + ((z1-z0-z2) << 48) + z0;
    return res;
}
ap_uint<96> karatsuba_48(ap_uint<48> x, 
                        ap_uint<48> y)
{
	if (x == 0 || y == 0) return 0;
    ap_uint<24> x_low = x(23, 0);
    ap_uint<24> x_high = x(47, 24);
    ap_uint<24> y_low = y(23, 0);
    ap_uint<24> y_high = y(47, 24);

    ap_uint<25> z1_left = x_low + x_high;
    ap_uint<25> z1_right = y_low + y_high;
    ap_uint<24> low1 = z1_left(23,0);
    ap_uint<24> low2 = z1_right(23,0);
    ap_uint<96> z0 = karatsuba_24(x_low, y_low);
    ap_uint<96> z2 = karatsuba_24(x_high, y_high);
    ap_uint<48> z1t = karatsuba_24(low1, low2);
    ap_uint<96> temp=z1_left[24]*z1_right[24];
    ap_uint<96> z1 = z1t+(ap_uint<48>(z1_left[24]*low2)<<24)+(ap_uint<48>(z1_right[24]*low1)<<24)+(temp<<48);
    ap_uint<96> res = (z2 << 48) + ((z1-z0-z2) << 24) + z0;
    return res;
}
ap_uint<48> karatsuba_24(ap_uint<24> x, 
                         ap_uint<24> y)
{
	if (x == 0 || y == 0) return 0;
    ap_uint<12> x_low = x(11, 0);
    ap_uint<12> x_high = x(23, 12);
    ap_uint<12> y_low = y(11, 0);
    ap_uint<12> y_high = y(23, 12);

    ap_uint<48> z0 = x_low*y_low;
    ap_uint<48> z2 = x_high*y_high;
    ap_uint<48> z1 = (x_low + x_high)*(y_low + y_high);
    ap_uint<48> res = (z2 << 24) + ((z1-z0-z2) << 12) + z0;
    return res;
}
uint384 ADD(uint384 a, 
            uint384 b, 
            uint384 q)
{
	uint384 temp = a + b;//a<2^381 & b<2^381
	if(temp>=q)
    {
		temp=temp-q;
    }
	return temp;
}
uint384 SUB(uint384 a, 
            uint384 b, 
            uint384 q)
{
	uint384 temp;
	if(a>=b)
    {
		temp = a - b;
	} 
    else 
    {
		temp = q - ( b - a );
	}
	return temp;
}
uint384 div(uint384 a, 
            uint384 b)
{
	uint384 temp=a/b;
	return temp;
}
uint384 barrett_reduce(uint768 a, 
                       uint384 q, 
                       uint768 mu)
{
    uint768 t = karatsuba_768(a, mu) >> 762;
    uint384 r = uint384(a - t * q);
    if (r >= q) 
    {
        r = r - q;
    }
    return r;
}
uint384 MUL(uint384 a, 
            uint384 b, 
            uint384 q, 
            uint384 INV)
{
	uint768 temp=karatsuba_384(a,b);
	uint384 res=montgomery_reduce(temp,q,INV);
	return res;
}
void identity(point* p, 
              int len)
{
	for(int i=0;i<len;i++)
	{
		p[i].x=0;
		p[i].y=1;
		p[i].z=0;
	}
	return;
}
point affine_to_project(uint384* p) //single point transform
{
	point res;
	res.x=p[0];
	res.y=p[1];
	res.z=1;
	return res;
}
int is_infinity(point p)
{
	if(p.x==0 && p.y==1 && p.z==0)
	{
		return 1;
	}
	return 0;
}
point PADD(const point& P1, 
           const point& P2, 
           const uint384& q, 
           const uint384& u) 
{
    if (P1.z == 0) 
    {
        return P2;
    }  // 
    if (P2.z == 0) 
    {
        return P1;
    }  // 
    uint384 Z1Z1 = MUL(P1.z, P1.z, q, u);    // Z1Z1 = Z1^2
    uint384 Z2Z2 = MUL(P2.z, P2.z, q, u);    // Z2Z2 = Z2^2
    uint384 U1 = MUL(P1.x, Z2Z2, q, u);      // U1 = X1*Z2Z2
    uint384 U2 = MUL(P2.x, Z1Z1, q, u);      // U2 = X2*Z1Z1
    uint384 S1 = MUL(P1.y, P2.z, q, u);
    S1 = MUL(S1, Z2Z2, q, u);                // S1 = Y1*Z2*Z2Z2
    uint384 S2 = MUL(P2.y, P1.z, q, u);
    S2 = MUL(S2, Z1Z1, q, u);                // S2 = Y2*Z1*Z1Z1
    uint384 H = SUB(U2, U1, q);              // H = U2-U1
    uint384 HH = MUL(H, H, q, u);            // HH = H^2
    uint384 I = ADD(H, H, q);
    I = MUL(I, I, q, u);                     // I = (2*H)^2
    uint384 J = MUL(H, I, q, u);             // J = H*I
    uint384 r = SUB(S2, S1, q);
    r = ADD(r, r, q);                        // r = 2*(S2-S1)
    uint384 V = MUL(U1, I, q, u);            // V = U1*I
    uint384 X3 = MUL(r, r, q, u);
    X3 = SUB(X3, J, q);
    X3 = SUB(X3, V, q);
    X3 = SUB(X3, V, q);                      // X3 = r^2 - J - 2*V
    uint384 Y3 = SUB(V, X3, q);
    Y3 = MUL(Y3, r, q, u);
    S1 = MUL(S1, J, q, u);                   // S1 = S1*J
    S1 = ADD(S1, S1, q);
    Y3 = SUB(Y3, S1, q);                     // Y3 = r*(V - X3) - 2*S1*J
    uint384 Z3 = ADD(P1.z, P2.z, q);
    Z3 = MUL(Z3, Z3, q, u);
    Z3 = SUB(Z3, Z1Z1, q);
    Z3 = SUB(Z3, Z2Z2, q);
    Z3 = MUL(Z3, H, q, u);                   // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H

    return {X3, Y3, Z3};
}

point PDBL(const point& P, 
           const uint384& q, 
           const uint384& u) 
{
    if (P.z == 0) 
    {
        return P;
    }
    uint384 XX = MUL(P.x, P.x, q, u);  // XX = X1^2
    uint384 YY = MUL(P.y, P.y, q, u);  // YY = Y1^2
    uint384 ZZ = MUL(P.z, P.z, q, u);  // ZZ = Z1^2
    uint384 S = MUL(P.x, YY, q, u);    // S = X1*YY
    S = ADD(S, S, q);                  // S = 2*X1*YY
    S = ADD(S, S, q);                  // S = 4*X1*YY

    uint384 M = ADD(XX, XX, q);        // M = 2*XX
    M = ADD(M, XX, q);                 

    uint384 T = MUL(M, M, q, u);       // T = M^2
    T = SUB(T, S, q);                  // T = M^2 - S
    T = SUB(T, S, q);                  // T = M^2 - 2*S

    uint384 Y3 = MUL(M, SUB(S, T, q), q, u); // Y3 = M*(S-T)
    YY = MUL(YY, YY, q, u);
    uint384 YY_eight = ADD(YY, YY, q); // 2*YY^2
    YY_eight = ADD(YY_eight, YY_eight, q); // 4*YY^2
    YY_eight = ADD(YY_eight, YY_eight, q); // 8*YY^2
    Y3 = SUB(Y3, YY_eight, q);         // Y3 = M*(S-T) - 8*YY^2

    uint384 Z3 = MUL(P.y, P.z, q, u);  // Z3 = Y1*Z1
    Z3 = ADD(Z3, Z3, q);               // Z3 = 2*Y1*Z1

    return {T, Y3, Z3};                
}
point PMUL(const point& P, 
           const uint256& k, 
           const uint384& q, 
           const uint384& u) 
{
    point result = {0, 1, 0};  
    PMUL_label0:for (int i = 256-1; i >= 0; --i) 
    { 
        result = PDBL(result, q, u);  
        if (k[i]==1) 
        {
            result = PADD(result, P, q, u);  
        }
    }
    return result;
}
epoint UniPadd(const epoint P, 
               const epoint Q, 
               uint384 k, 
               uint384 q, 
               uint384 INV)
{ //377
#pragma HLS PIPELINE II=3
//k=d'/2
	if (P.z == 0) 
    {
        return Q;
    }
	if (Q.z == 0)
    {
        return P;
    }
	// Step 1: Calculate (Y1 - X1) and (Y2 - X2)
    uint384 Y1_minus_X1 = SUB(P.y, P.x, q);
    uint384 Y2_minus_X2 = SUB(Q.y, Q.x, q);
    // Step 2: Calculate A = (Y1 - X1) * (Y2 - X2)
    uint384 A = MUL(Y1_minus_X1, Y2_minus_X2, q, INV);
    // Step 3: Calculate (Y1 + X1) and (Y2 + X2)
    uint384 Y1_plus_X1 = ADD(P.y, P.x, q);
    uint384 Y2_plus_X2 = ADD(Q.y, Q.x, q);
    // Step 4: Calculate B = (Y1 + X1) * (Y2 + X2)
    uint384 B = MUL(Y1_plus_X1, Y2_plus_X2, q, INV);
    // Step 5: Calculate C = k * T1 * T2
    uint384 T1_times_T2 = MUL(P.t, Q.t, q, INV);
    uint384 C = MUL(k, T1_times_T2, q, INV);
    // Step 6: Calculate D = 2 * Z1 * Z2
    uint384 Z1_times_Z2 = MUL(P.z, Q.z, q, INV);
    uint384 D = ADD(Z1_times_Z2, Z1_times_Z2, q);  // D = 2 * Z1 * Z2
    // Step 7: Calculate E = B - A
    uint384 E = SUB(B, A, q);
    // Step 8: Calculate F = D - C
    uint384 F = SUB(D, C, q);
    // Step 9: Calculate G = D + C
    uint384 G = ADD(D, C, q);
    // Step 10: Calculate H = B + A
    uint384 H = ADD(B, A, q);
    // Step 11: Calculate the result coordinates
    epoint R;
    R.x = MUL(E, F, q, INV);            // X3 = E * F
    R.y = MUL(G, H, q, INV);            // Y3 = G * H
    R.t = MUL(E, H, q, INV);            // T3 = E * H
    R.z = MUL(F, G, q, INV);            // Z3 = F * G
    return R;
}
point ezmul(const point* pos, 
            const uint256* scalars, 
            const uint384& q, 
            const uint384& u)
{
	point temp;
	point res={0,1,0};
	for(int i=0;i<SCALAR_NUM;i++)
	{
		temp=PMUL(pos[i],scalars[i],q,u);
		res=PADD(res,temp,q,u);
	}
	return res;
}
void initial(point buckets[WINDOW_NUM][BUCKET_NUM])
{
	initial_label0:for(int i=0;i<WINDOW_NUM;i++)
	{
		initial_label1:for(int j=0;j<BUCKET_NUM;j++)
		{
			buckets[i][j]={0,1,0};
		}
	}
}
void bucket_part(const point p, 
                 const uint256 s, 
                 point buckets[WINDOW_NUM][BUCKET_NUM],
                 const uint384& q,
                 const uint384& inv)
{
	point ptemp;
	ptemp.x=p.x;
	ptemp.y=q-p.y;
	ptemp.z=p.z;

	uint256 temps=s;
	for(int i=0;i<WINDOW_NUM;i++)
	{
		uint256 temp=temps((i+1)*WINDOW_SIZE-1,i*WINDOW_SIZE);
		if(temp[WINDOW_SIZE-1]==1)
		{
			temp=(1<<WINDOW_SIZE)-temp;
			temps+=1<<((i+1)*WINDOW_SIZE);
			buckets[i][temp]=PADD(buckets[i][temp],ptemp,q,inv);
		}
		else{
			buckets[i][temp]=PADD(buckets[i][temp],p,q,inv);
		}
	}
}
void normal_part(const point p, 
                 const uint256 s, 
                 point buckets[WINDOW_NUM][BUCKET_NUM],
                 const uint384& q,const uint384& inv)
{
	normal_part_label2:for(int i=0;i<WINDOW_NUM;i++)
	{
		int stemp=s(WINDOW_SIZE*(i+1)-1,WINDOW_SIZE*i);
		buckets[i][stemp]=PADD(buckets[i][stemp],p,q,inv);
	}
}
void bucket_part_new(const point p, 
                     const uint256 s, 
                     point buckets[WINDOW_NUM][BUCKET_NUM],
                     const uint384& q,
                     const uint384& inv)
{
	point ptemp;
	ptemp.x=p.x;
	ptemp.y=q-p.y;
	ptemp.z=p.z;
	ap_uint<WINDOW_SIZE+1> s_ary[WINDOW_NUM]; //a_i true num
	uint256 temps=s;
	for(int i=0;i<WINDOW_NUM;i++)
	{
		ap_uint<WINDOW_SIZE+1> temp=temps((i+1)*WINDOW_SIZE-1,i*WINDOW_SIZE);
		if(temp[WINDOW_SIZE-1]==1)
		{
			temp=ONE<<WINDOW_SIZE-temp;
			temps+=ONE<<((i+1)*WINDOW_SIZE);
		}
		s_ary[i]=temp;
	}

	bucket_part_new_label0:for(int i=0;i<WINDOW_NUM;i++)
	{
		int indext=s_ary[i];
		if(indext<=0) //how to notice points' change ->use inv
		{
			buckets[i][indext-1]=PADD(buckets[i][indext-1],ptemp,q,inv);
		}
		else
		{
			buckets[i][indext-1]=PADD(buckets[i][indext-1],p,q,inv);
		} 
	}
}
void bucket_accum(point buckets[WINDOW_NUM][BUCKET_NUM],
                  point bus[WINDOW_NUM],
                  const uint384& q,
                  const uint384& inv)
{
	for(int i=0;i<WINDOW_NUM;i++)
	{
		point tmp={0,1,0};
		point tmp1={0,1,0};
		for(int j=BUCKET_NUM-1;j>0;j--)
		{
			tmp=PADD(tmp,buckets[i][j],q,inv);
			tmp1=PADD(tmp1,tmp,q,inv);
		}
		bus[i]=tmp1;
	}
}
void allpart(const point* p, 
             const uint256* s, 
             point buckets[WINDOW_NUM][BUCKET_NUM], 
             const uint384& q, 
             const uint384& inv)
{
	point buckets2[WINDOW_NUM][BUCKET_NUM];
	point buckets3[WINDOW_NUM][BUCKET_NUM];
	point buckets4[WINDOW_NUM][BUCKET_NUM];
	point buckets5[WINDOW_NUM][BUCKET_NUM];
	point buckets6[WINDOW_NUM][BUCKET_NUM];
	allpart_label0:for(int i=0;i<1048;i+=6)
	{
		//bucket_part(pos[i],scalars[i],buckets,q,inv);
		bucket_part_new(p[i],s[i],buckets,q,inv);
		bucket_part_new(p[i+1],s[i+1],buckets2,q,inv);
		bucket_part_new(p[i+2],s[i+2],buckets3,q,inv);
		bucket_part_new(p[i+3],s[i+3],buckets4,q,inv);
		bucket_part_new(p[i+4],s[i+4],buckets5,q,inv);
		bucket_part_new(p[i+5],s[i+5],buckets6,q,inv);
		//normal_part(p,s,buckets,q,inv);
	}
}
void bucket_accum(point buckets[WINDOW_NUM][BUCKET_NUM],
                  point bus[WINDOW_NUM],
                  const uint384& q,
                  const uint384& inv)
{
	for(int i=0;i<WINDOW_NUM;i++)
	{
		point tmp={0,1,0};
		point tmp1={0,1,0};
		for(int j=BUCKET_NUM-1;j>0;j--)
		{
			tmp=PADD(tmp,buckets[i][j],q,inv);
			tmp1=PADD(tmp1,tmp,q,inv);
		}
		bus[i]=tmp1;
	}
}
point bucket_aggre(point bus[WINDOW_NUM],
                   const uint384& q,
                   const uint384& inv)
{
	point res={0,1,0};
	for(int i=0;i<WINDOW_NUM;i++)
	{
		point temp;
		temp.x=bus[i].x;
		temp.y=bus[i].y;
		temp.z=bus[i].z;
		for(int j=0;j<i*WINDOW_SIZE;j++)
		{
			temp=PDBL(temp,q,inv);
		}
		res=PADD(res,temp,q,inv);
	}
	return res;
}
point bucket_aggre_new(point bus[WINDOW_NUM],
                       const uint384& q,
                       const uint384& inv)
{
	point res={0,1,0};
	for(int i=WINDOW_NUM-1;i>0;i--)
	{
		point temp;
		temp.x=bus[i].x;
		temp.y=bus[i].y;
		temp.z=bus[i].z;
		temp=PADD(res,temp,q,inv);
		for(int j=0;j<8;j++)
		{
			res=PDBL(temp,q,inv);
		}
	}
	point temp1;
	temp1.x=bus[0].x;
	temp1.y=bus[0].y;
	temp1.z=bus[0].z;
	res=PADD(res,temp1,q,inv);
	return res;
}
point bucket_aggre_new2(point bus[WINDOW_NUM], 
                        const uint384& q, 
                        const uint384& inv) {//correct and better
    point res = {0, 1, 0};
    point temp[WINDOW_NUM];
    for (int i = 0; i < WINDOW_NUM; i++) {
        temp[i].x = bus[i].x;
        temp[i].y = bus[i].y;
        temp[i].z = bus[i].z;
    }
    bucket_aggre_new_label1:for (int i = WINDOW_NUM - 1; i > 0; i--) {
        res = PADD(res, temp[i], q, inv);
        bucket_aggre_new_label0:for (int j = 0; j < WINDOW_SIZE; j++) {
            res = PDBL(res, q, inv);
        }
    }
    res = PADD(res, temp[0], q, inv);
    return res;
}
point msm_pippenger(const point pos[SCALAR_NUM], 
                    const uint256 scalars[SCALAR_NUM], 
                    const uint384& q, 
                    const uint384& inv)
{
	point buckets[WINDOW_NUM][BUCKET_NUM];
	initial(buckets);
	for(int i=0;i<SCALAR_NUM;i++)
	{
		bucket_part(pos[i],scalars[i],buckets,q,inv);
	}
	point bus[WINDOW_NUM];
	bucket_accum(buckets,bus,q,inv);
	point res;
	res=bucket_aggre_new(bus,q,inv);
	return res;
}
